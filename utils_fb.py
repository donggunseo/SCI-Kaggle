import pandas as pd
from tqdm import tqdm
import numpy as np
from prepare import label_dict

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

def calc_overlap(row):
    """
    Calculates the overlap between prediction and
    ground truth and overlap percentages used for determining
    true positives.
    """
    set_pred = set(row.predictionstring_pred.split(' '))
    set_gt = set(row.predictionstring_gt.split(' '))
    # Length of each and intersection
    len_gt = len(set_gt)
    len_pred = len(set_pred)
    inter = len(set_gt.intersection(set_pred))
    overlap_1 = inter / len_gt
    overlap_2 = inter/ len_pred
    return [overlap_1, overlap_2]


def score_feedback_comp(pred_df, gt_df):
    """
    A function that scores for the kaggle
        Student Writing Competition
        
    Uses the steps in the evaluation page here:
        https://www.kaggle.com/c/feedback-prize-2021/overview/evaluation
    """
    gt_df = gt_df[['id','discourse_type','predictionstring']] \
        .reset_index(drop=True).copy()
    pred_df = pred_df[['id','class','predictionstring']] \
        .reset_index(drop=True).copy()
    pred_df['pred_id'] = pred_df.index
    gt_df['gt_id'] = gt_df.index
    # Step 1. all ground truths and predictions for a given class are compared.
    joined = pred_df.merge(gt_df,
                           left_on=['id','class'],
                           right_on=['id','discourse_type'],
                           how='outer',
                           suffixes=('_pred','_gt')
                          )
    joined['predictionstring_gt'] = joined['predictionstring_gt'].fillna(' ')
    joined['predictionstring_pred'] = joined['predictionstring_pred'].fillna(' ')

    joined['overlaps'] = joined.apply(calc_overlap, axis=1)

    # 2. If the overlap between the ground truth and prediction is >= 0.5, 
    # and the overlap between the prediction and the ground truth >= 0.5,
    # the prediction is a match and considered a true positive.
    # If multiple matches exist, the match with the highest pair of overlaps is taken.
    joined['overlap1'] = joined['overlaps'].apply(lambda x: eval(str(x))[0])
    joined['overlap2'] = joined['overlaps'].apply(lambda x: eval(str(x))[1])


    joined['potential_TP'] = (joined['overlap1'] >= 0.5) & (joined['overlap2'] >= 0.5)
    joined['max_overlap'] = joined[['overlap1','overlap2']].max(axis=1)
    tp_pred_ids = joined.query('potential_TP') \
        .sort_values('max_overlap', ascending=False) \
        .groupby(['id','predictionstring_gt']).first()['pred_id'].values

    # 3. Any unmatched ground truths are false negatives
    # and any unmatched predictions are false positives.
    fp_pred_ids = [p for p in joined['pred_id'].unique() if p not in tp_pred_ids]

    matched_gt_ids = joined.query('potential_TP')['gt_id'].unique()
    unmatched_gt_ids = [c for c in joined['gt_id'].unique() if c not in matched_gt_ids]

    # Get numbers of each type
    TP = len(tp_pred_ids)
    FP = len(fp_pred_ids)
    FN = len(unmatched_gt_ids)
    #calc microf1
    my_f1_score = TP / (TP + 0.5*(FP+FN))
    return my_f1_score

def postprocess_fb_predictions(
    eval_examples,
    eval_datasets,
    predictions,
):
    print(predictions.shape)
    preds = np.argmax(predictions, axis=-1)
    i2l, l2i, N_LABELS = label_dict()
    all_prediction = []
    for k, label_pred in tqdm(enumerate(preds)):
      token_preds = [i2l[i] for i in label_pred]
      each_prediciton = []
      word_ids = eval_datasets['word_ids'][k]
      previous_word_idx = -1
      for idx, word_idx in enumerate(word_ids):
        if word_idx == None:
          continue
        elif word_idx != previous_word_idx:
          each_prediciton.append(token_preds[idx])
          previous_word_idx = word_idx
      all_prediction.append(each_prediciton)
    final_pred = []
    for i in range(len(eval_datasets)):
      idx = eval_datasets['id'][i]
      pred = all_prediction[i]
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
        if cls != 'O' and cls != '' and end-j>7:
            final_pred.append((idx, cls.replace('I-', ''), ' '.join(map(str, list(range(j, end))))))
        j = end
    oof = pd.DataFrame(final_pred)
    oof.columns = ['id', 'class', 'predictionstring']
    
    oof = link_evidence(oof)
    return oof