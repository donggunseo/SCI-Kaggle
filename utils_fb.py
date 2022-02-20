import pandas as pd
from tqdm import tqdm
import numpy as np
from prepare import label_dict
import torch
import random
import os

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def link_evidence(oof):
  if not len(oof):
    return oof
  
  def jn(pst, start, end):
    return " ".join([str(x) for x in pst[start:end]])
  
  thresh = 1
  idu = oof['id'].unique()
  eoof = oof[oof['class'] == "Evidence"]
  neoof = oof[oof['class'] != "Evidence"]
  eoof.index = eoof[['id', 'class']]
  for thresh2 in range(26, 27, 1):
    retval = []
    for idv in tqdm(idu, desc='link_evidence', leave=False):
      for c in ['Evidence']:
        q = eoof[(eoof['id'] == idv)]
        if len(q) == 0:
          continue
        pst = []
        for r in q.itertuples():
          pst = [*pst, -1,  *[int(x) for x in r.predictionstring.split()]]
        start = 1
        end = 1
        for i in range(2, len(pst)):
          cur = pst[i]
          end = i
          if  ((cur == -1) and ((pst[i + 1] > pst[end - 1] + thresh) or (pst[i + 1] - pst[start] > thresh2))):
            retval.append((idv, c, jn(pst, start, end)))
            start = i + 1
        v = (idv, c, jn(pst, start, end + 1))
        retval.append(v)
    roof = pd.DataFrame(retval, columns=['id', 'class', 'predictionstring'])
    roof = roof.merge(neoof, how='outer')
    return roof

def find_max_label(word_prediction_score):
    x = np.sum(word_prediction_score, axis=0)
    max_label = np.argmax(x, axis=-1)
    return max_label

def postprocess_fb_predictions2(
    eval_datasets,
    predictions,
):
    proba_thresh = {
        "Lead": 0.687,
        "Position": 0.537,
        "Evidence": 0.637,
        "Claim": 0.537,
        "Concluding Statement": 0.687,
        "Counterclaim": 0.537,
        "Rebuttal": 0.537,
    }
    #discourse length threshold
    min_thresh = {
        "Lead": 9,
        "Position": 5,
        "Evidence": 14,
        "Claim": 3,
        "Concluding Statement": 11,
        "Counterclaim": 6,
        "Rebuttal": 4,
    }
    print(predictions.shape)
    softmax = torch.nn.Softmax(dim=-1)
    predictions = torch.tensor(predictions)
    pred_score = softmax(predictions)
    pred_score = pred_score.numpy()
    i2l, _, _ = label_dict()
    all_prediction = []
    all_pred_score=[]
    for k, label_pred_score in tqdm(enumerate(pred_score), desc = "post-processing"):
      each_prediction = []
      each_prediction_score = []
      word_ids = eval_datasets['word_ids'][k]
      previous_word_idx = -1
      word_prediction_score=[]
      for idx, word_idx in enumerate(word_ids):
        if word_idx == None:
          continue
        
        elif word_idx != previous_word_idx:
          if len(word_prediction_score)!=0:
            # find label which have the most score label following each tokens including in one word
            max_label = find_max_label(word_prediction_score)
            word_prediction_score = [word_prediction_score[i][max_label] for i in range(len(word_prediction_score))]
            each_prediction_score.append(word_prediction_score)
            each_prediction.append(i2l[max_label])
          
          previous_word_idx = word_idx
          word_prediction_score=[]
          word_prediction_score.append(label_pred_score[idx])
        
        else:
          word_prediction_score.append(label_pred_score[idx])
      
      max_label = find_max_label(word_prediction_score)
      word_prediction_score = [word_prediction_score[i][max_label] for i in range(len(word_prediction_score))]  
      each_prediction_score.append(word_prediction_score)
      each_prediction.append(i2l[max_label])
      
      all_prediction.append(each_prediction)
      all_pred_score.append(each_prediction_score)
    final_pred = []
    for i in range(len(eval_datasets['id'])):
      idx = eval_datasets['id'][i]
      pred = all_prediction[i]
      pred_score = all_pred_score[i]
      j=0
      while j < len(pred):
        cls = pred[j]
        if cls =='O': 
            j+=1
        else: 
            cls = cls.replace('B', 'I')
        end = j+1
        while end < len(pred) and pred[end] == cls:
            end +=1
        final_pred_score = []
        for item in pred_score[j:end]:
          final_pred_score.extend(item)
        if cls != 'O' and cls!='' and sum(final_pred_score)/len(final_pred_score)>=proba_thresh[cls.replace('I-', '')] and len(final_pred_score)>=min_thresh[cls.replace('I-', '')]:
            final_pred.append((idx, cls.replace('I-', ''), ' '.join(map(str, list(range(j, end))))))
        j = end
    oof = pd.DataFrame(final_pred)
    oof.columns = ['id', 'class', 'predictionstring']
    
    oof = link_evidence(oof)
    return oof

def calc_overlap3(set_pred, set_gt):
    """
    Calculates if the overlap between prediction and
    ground truth is enough fora potential True positive
    """
    # Length of each and intersection
    try:
        len_gt = len(set_gt)
        len_pred = len(set_pred)
        inter = len(set_gt & set_pred)
        overlap_1 = inter / len_gt
        overlap_2 = inter/ len_pred
        return overlap_1 >= 0.5 and overlap_2 >= 0.5
    except:  # at least one of the input is NaN
        return False

def score_feedback_comp_micro3(pred_df, gt_df, discourse_type):
    """
    A function that scores for the kaggle
        Student Writing Competition
        
    Uses the steps in the evaluation page here:
        https://www.kaggle.com/c/feedback-prize-2021/overview/evaluation
    """
    gt_df = gt_df.loc[gt_df['discourse_type'] == discourse_type, 
                      ['id', 'predictionstring']].reset_index(drop=True)
    pred_df = pred_df.loc[pred_df['class'] == discourse_type,
                      ['id', 'predictionstring']].reset_index(drop=True)
    pred_df['pred_id'] = pred_df.index
    gt_df['gt_id'] = gt_df.index
    pred_df['predictionstring'] = [set(pred.split(' ')) for pred in pred_df['predictionstring']]
    gt_df['predictionstring'] = [set(pred.split(' ')) for pred in gt_df['predictionstring']]
    
    # Step 1. all ground truths and predictions for a given class are compared.
    joined = pred_df.merge(gt_df,
                          left_on='id',
                          right_on='id',
                          how='outer',
                          suffixes=('_pred','_gt')
                          )
    overlaps = [calc_overlap3(*args) for args in zip(joined.predictionstring_pred, 
                                                    joined.predictionstring_gt)]
    
    # 2. If the overlap between the ground truth and prediction is >= 0.5, 
    # and the overlap between the prediction and the ground truth >= 0.5,
    # the prediction is a match and considered a true positive.
    # If multiple matches exist, the match with the highest pair of overlaps is taken.
    # we don't need to compute the match to compute the score
    TP = joined.loc[overlaps]['gt_id'].nunique()

    # 3. Any unmatched ground truths are false negatives
    # and any unmatched predictions are false positives.
    TPandFP = len(pred_df)
    TPandFN = len(gt_df)
    
    #calc microf1
    my_f1_score = 2*TP / (TPandFP + TPandFN)
    return my_f1_score

def score_feedback_comp3(pred_df, gt_df, return_class_scores=False):
    class_scores = {}
    for discourse_type in gt_df.discourse_type.unique():
        class_score = score_feedback_comp_micro3(pred_df, gt_df, discourse_type)
        class_scores[discourse_type] = class_score
    f1 = np.mean([v for v in class_scores.values()])
    if return_class_scores:
        return f1, class_scores
    return f1