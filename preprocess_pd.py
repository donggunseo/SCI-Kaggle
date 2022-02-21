import re
import pandas as pd
import numpy as np
from tqdm import tqdm
#from datasets import Dataset

PUNCTUATION = set(".,;")

BASE_DIR = "../input/feedback-prize-2021/"
TRAIN_DIR = BASE_DIR + 'train'

df = pd.read_csv(BASE_DIR + "train.csv")

def get_new_positions(ids):
    """
    correction #1 : new_start, new_end
    """
    new_starts = []
    new_ends = []
    new_texts = []
  
    for id_ in tqdm(ids):
        
        with open(f"{TRAIN_DIR}/{id_}.txt",'r', encoding = 'utf-8') as fp:
            file_text = fp.read()

        discourse_data = df[df["id"] == id_]

        discourse_texts = discourse_data["discourse_text"]
        discourse_starts = discourse_data["discourse_start"]
        for disc_text, disc_start in zip(discourse_texts, discourse_starts):
            disc_text = disc_text.strip()

            matches = [x for x in re.finditer(re.escape(disc_text), file_text)]
            
            # disc_text가 file_text와 겹치는 파트를 iter object로 반환
            offset = 0
            while len(matches) == 0 and offset < len(disc_text):
                # disc_text string 통째로에 대해 match되는 게 없는 경우 들어오게 되는 if문. (discourse_text가 문단인 경우가 이렇게 될 수 있음.)
                # 여기로 들어오는 사례가 딱 한번밖에 없음 (ID: F91D7BB4277C)
                # 그 외엔 discourse_text가 txt file에 모두 그대로 있음.
                chunk = disc_text if offset == 0 else disc_text[:-offset]
                matches = [x for x in re.finditer(re.escape(chunk), file_text)]
                offset += 5
            if offset >= len(disc_text):
                # 여기로 들어오는 경우는 없음
                print(f"Could not find substring in {disc_text}")
                print(matches)
                continue

            # There are some instances when there are multiple matches, 
            # so we'll take the closest one to the original discourse_start
            distances = [abs(disc_start-match.start()) for match in matches]

            idx = matches[np.argmin(distances)].start()                 # 시작점은 txt file index를 기준으로

            end_idx = idx + len(disc_text)          # 끝점은 disc_text 길이를 기준으로 맞추기

            final_text = file_text[idx:end_idx]
            
            new_starts.append(idx)
            new_ends.append(idx + len(final_text))
            new_texts.append(final_text)
            
    return {
        "new_start": new_starts,
        "new_end": new_ends,
        "text_by_new_index": new_texts
    }

def get_new_predstr(ids):
    """
    correction #2 : predictionstring
    """
    new_pred_strings = []
    
    for id_ in tqdm(ids):
        with open(f"{TRAIN_DIR}/{id_}.txt",'r', encoding = 'utf-8') as fp:
            file_text = fp.read()

        discourse_data = df[df["id"] == id_]
        
        left_idxs = discourse_data["new_start"]
        right_idxs = discourse_data["new_end"]
        
        for left_idx, right_idx in zip(left_idxs, right_idxs):
            start_word_id = len(file_text[:left_idx].split())
            
            if left_idx > 0 and file_text[left_idx].split() != [] and file_text[left_idx-1].split() != []:
                start_word_id -= 1
                
            end_word_id = start_word_id + len(file_text[left_idx:right_idx].split())
            
            new_pred_strings.append(" ".join(list(map(str, range(start_word_id, end_word_id)))))
            
    return {
        "new_predictionstring": new_pred_strings
    }

def preprocess():       

    id_df = df['id'].unique()
    print(id_df)
    # correction #1 : new_start, new_end
    results = get_new_positions(id_df)
    df["new_start"] = results["new_start"]
    df["new_start"] = results["new_start"]
    df["new_end"] = results["new_end"]
    df["new_discourse_text"] = results["text_by_new_index"]

    # correction #2 : predictionstring 
    results = get_new_predstr(id_df)
    df["new_predictionstring"] = results["new_predictionstring"]

    df.drop(['discourse_start', 'discourse_end', 'discourse_text', 'predictionstring'], axis='columns', inplace=True)
    df.rename(columns = {'new_start':'discourse_start', 'new_end':'discourse_end', 'new_discourse_text':'discourse_text', 'new_predictionstring':'predictionstring'}, inplace=True)
    # print(" change predictionstring: ", len(df[df.new_predictionstring != df.predictionstring]))
    # print(" change start/end: ", len(df[(df.new_start != df.discourse_start) | (df.new_end != df.discourse_end)]))
    return df
