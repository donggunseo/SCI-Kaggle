from autocorrect import Speller
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import random
import re
from numpy import dot
from numpy.linalg import norm
import nltk
from utils_fb import seed_everything
from joblib import Parallel, delayed
nltk.download('averaged_perceptron_tagger')

BASE_DIR = "../input/feedback-prize-2021/"
TRAIN_DIR = BASE_DIR + 'train'
SAVE_DIR = BASE_DIR + 'train_oversamples'
DISCOURSE_TYPES = ['Rebuttal', 'Counterclaim', 'Lead', 'Concluding Statement', 'Claim', 'Position', 'Evidence']
WEAK_DISCOURSE_TYPES = ['Rebuttal', 'Counterclaim']
FULL_DISCOURSE_TYPES = ['Lead', 'Concluding Statement', 'Claim', 'Position', 'Evidence']
GLOVE_PATH = '../glove/glove.6B/glove.6B.50d.txt'

def open_txt():
    train_names, train_texts = [], []
    for f in tqdm(list(os.listdir(TRAIN_DIR))):
        train_names.append(f.replace('.txt', ''))
        train_texts.append(open('../input/feedback-prize-2021/train/' + f, 'r', encoding='utf-8').read())
    train_text_df = pd.DataFrame({'id': train_names, 'text': train_texts})
    return train_text_df

def train_ids2discourse_type_counts(X_ids, id2types):
    no_num = re.compile('[^0-9]')
    type_count = dict([(dt, 0) for dt  in DISCOURSE_TYPES])
    for _id in X_ids:
        for dt in id2types[_id]:
            dt_name = "".join(no_num.findall(dt)).rstrip(" ")
            type_count[dt_name] += 1
    return pd.Series(type_count).sort_values()

def correct_misspelling(train_text):
    correct_count = 0
    spell = Speller(lang='en')
    for idx, word in enumerate(train_text):
        cor_word = spell(word)
        if word != cor_word:
            train_text[idx] = cor_word
            correct_count += 1
    return train_text, correct_count

def most_similar(glove_model, word):
    a = glove_model[word]
    best = (None, 0.0)
    for w, b in glove_model.items():
        value = dot(a, b)/(norm(a)*norm(b))
        if w != word and value > best[1]:
            best = (w, value)
    return best[0]

def do_change(glove_model, example, txt):
    PUNCTUATION = set([".",",",";"])
    ALLOW_POS_TAGS = ['JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNPS', 'RBR', 'RBS', 'VB', 'VBD', 'VBG']
    
    discourse_text = example['discourse_text'].split()
    predictionstring = example['predictionstring'].split()
    # train_text = example['text'].split()
    train_text = txt.split()
    
    if len(discourse_text) != len(predictionstring):
        print("mismatch! so cannot oversampling this discorse type")
        return 
    
    # correct misspelling
    train_text, correction_count = correct_misspelling(train_text)
    
    # synonym replacement
    
    is_replaced = False
    while not is_replaced:
        list_idx = int(np.random.choice(len(predictionstring), 1))
        txt_idx = int(predictionstring[list_idx])
        origin_word = discourse_text[list_idx]
        
        pos_tag = nltk.pos_tag([origin_word])[0][1]
        if PUNCTUATION & set(list(origin_word)) or (origin_word not in glove_model.keys()) or (pos_tag not in ALLOW_POS_TAGS):
            continue
        
        replace_word = most_similar(glove_model, origin_word)
        
        train_text[txt_idx] = replace_word
        discourse_text[list_idx] = replace_word
        is_replaced = True
            
    example['predictionstring'] = example['predictionstring']
    example['discourse_text'] = " ".join(discourse_text)
    txt = " ".join(train_text)
    
    return example, txt, correction_count

def oversampling(glove_model, train_text_df, add_ids, df):
    new_df = pd.DataFrame(columns=df.columns)
    for id in tqdm(add_ids):
        count = 0
        new_id = "{0}_S".format(id)
        
        examples = df[df.id == id]          # id의 모든 discourse type 행을 추출
        text = train_text_df[train_text_df.id == id]['text'].values[0]  # id에 맞는 원본 text를 로드
        
        for i, example in examples.iterrows(): 
            # id내 annotation된 discourse type을 살피면서 워드를 바꾼다.
            # Rebuttal, Counterclaim인 경우, 무조건 바꿈
            # 그외 타입인 경우, 확률적으로 바꿈.
            #
            # discourse type이 변경된 경우, df에 계속해서 추가한다.
            # text는 모든 변경사항을 누적시키며, for문이 끝난 후 새로운 파일로 저장함.
            new_example = example.copy()
            if example['discourse_type'] in WEAK_DISCOURSE_TYPES:
                new_example, text, correction_count = do_change(glove_model, new_example, text)
                count += 1
            else:
                if random.random() > 0.8:
                    new_example, text, correction_count = do_change(glove_model, new_example, text)
                    count += 1
            new_example['id'] = new_id
            new_df = pd.concat([new_df, new_example], ignore_index=True)
        # save txt 
        with open(SAVE_DIR+ '/{0}.txt'.format(new_id), 'w', encoding = 'utf-8') as f:
            f.write(text)
        print("new_id: {0} \t Add {2} discourse types. (# of synonym raplacement: {3}, # of correction misspelling: {4}".format(new_id, len(new_df), len(examples), count, correction_count))     
    print("complete oversampling --  total new rows : ", len(new_df))         
    return new_df

if __name__ == "__main__":
    seed_everything(42)
    if not os.path.exists(SAVE_DIR):
        os.mkdir(SAVE_DIR)
    df = pd.read_csv(BASE_DIR + "train_corrected.csv")  
    train_text_df = open_txt()
    X_ids = list(df['id'].unique())
    id2types = df.groupby('id')['discourse_type'].unique().to_dict()
    type_count = train_ids2discourse_type_counts(X_ids, id2types)
    FILL_TO = max(type_count)
    add_ids = []
    for dt in tqdm(WEAK_DISCOURSE_TYPES):
        print(dt)
        # Get current Discourse Type Count
        type_count = train_ids2discourse_type_counts(X_ids, id2types)
        dt_sample_count = type_count[dt]
        if dt_sample_count < FILL_TO:
            while dt_sample_count < FILL_TO:
                # Take Random ID
                random_id = str(np.random.choice(X_ids, 1).squeeze())
                if dt in id2types[random_id] :
                    X_ids.append(random_id)
                    add_ids.append(random_id)
                    dt_sample_count += 1
    print("sampling id count :", len(add_ids))
    print("MAX count :", FILL_TO)
    with open(GLOVE_PATH, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    glove_model = {}
    for line in tqdm(lines):
        split_line = line.split()
        word = split_line[0]
        embedding = np.array(split_line[1:], dtype=np.float64)
        glove_model[word] = embedding
    print(f"{len(glove_model)} words loaded!")
    new_df = df.copy()
    print(len(new_df))
    new_df_list = np.array_split(new_df, 80)
    new_df_list = Parallel(n_jobs = 80, backend = 'multiprocessing')(delayed(oversampling)(glove_model, train_text_df, add_ids, temp_df) for temp_df in new_df_list)
    new_df = pd.concat(new_df_list, ignore_index=True)
    new_df.to_csv('train_oversampled.csv', index=False)
    combined_df = pd.concat([df, new_df], ignore_index=True)
    combined_df.to_csv('train_corrected_oversampled.csv', index=False)