import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from datasets import Dataset
import os
from create_kfold import create_kfold
from transformers import AutoTokenizer, DataCollatorForTokenClassification

def label_dict():
    train = pd.read_csv('../input/feedback-prize-2021/train.csv')
    classes = train.discourse_type.unique().tolist()
    tags = defaultdict()
    for i, c in enumerate(classes):
        tags[f'B-{c}'] = i
        tags[f'I-{c}'] = i + len(classes)
    tags[f'O'] = len(classes) * 2
    tags[f'Special'] = -100
    l2i = dict(tags)
    i2l = defaultdict()
    for k, v in l2i.items(): 
        i2l[v] = k
    i2l[-100] = 'Special'
    i2l = dict(i2l)
    N_LABELS = len(i2l) - 1 # not accounting for -100
    return i2l, l2i, N_LABELS

def prepare_datasets(kfold = 5):
    #read csv
    train = pd.read_csv('../input/feedback-prize-2021/train.csv')
    #kfold csv 
    train_kfold, train = create_kfold(df=train, k=kfold)
    #make label_to_id and id_to_label dict
    
    i2l, l2i, N_LABELS = label_dict()

    #make csv for full text file
    train_names, train_texts = [], []
    for f in tqdm(list(os.listdir('../input/feedback-prize-2021/train'))):
        train_names.append(f.replace('.txt', ''))
        train_texts.append(open('../input/feedback-prize-2021/train/' + f, 'r', encoding='utf-8').read())
    train_text_df = pd.DataFrame({'id': train_names, 'text': train_texts})
    train_text_df = train_text_df.merge(train_kfold, on="id", how="left")
    
    # make NER label for each word sperated by split()
    all_entities = []
    all_text_list = []
    for ii,i in tqdm(enumerate(train_text_df.iterrows())):
        text_list = i[1]['text'].split()
        total = text_list.__len__()
        entities = ["O"]*total
        for j in train[train['id'] == i[1]['id']].iterrows():
            discourse = j[1]['discourse_type']
            list_ix = [int(x) for x in j[1]['predictionstring'].split(' ')]
            entities[list_ix[0]] = f"B-{discourse}"
            for k in list_ix[1:]: entities[k] = f"I-{discourse}"
        all_entities.append(entities)
        all_text_list.append(text_list)
    train_text_df['entities'] = all_entities
    train_text_df['text_list'] = all_text_list
    
    tokenizer = AutoTokenizer.from_pretrained('allenai/longformer-base-4096', add_prefix_space=True)
    
    def fix_beginnings(labels):
        for i in range(1,len(labels)):
            curr_lab = labels[i]
            prev_lab = labels[i-1]
            if curr_lab in range(7,14):
                if prev_lab != curr_lab and prev_lab != curr_lab - 7:
                    labels[i] = curr_lab -7
        return labels
    
    def preparing_train_dataset(examples):
        encoding = tokenizer(examples['text_list'], truncation=True, padding=False, max_length = 2048, is_split_into_words=True)
        total= len(encoding['input_ids'])
        encoding['word_ids']=[]
        encoding['labels']=[]
        encoding['id']=[]
        for i in range(total):
            labels = [l2i['O'] for _ in range(len(encoding['input_ids'][i]))]
            word_idx = encoding.word_ids(batch_index=i)
            for j in range(len(word_idx)):
                if word_idx[j] is None:
                    labels[j]=l2i['Special']
                else:
                    label = examples['entities'][i][word_idx[j]]
                    labels[j]=l2i[label]
            labels = fix_beginnings(labels)
            encoding['labels'].append(labels)
            encoding['word_ids'].append(word_idx)
            encoding['id'].append(examples['id'][i])
        return encoding
    
    kfold_tokenized_datasets = []
    kfold_examples = []
    for i in range(kfold):
        df = train_text_df[train_text_df['kfold']==i].reset_index(drop=True)
        datasets = Dataset.from_pandas(df)
        tokenized_datasets = datasets.map(preparing_train_dataset, batched=True, batch_size=1000, remove_columns=datasets.column_names)
        kfold_tokenized_datasets.append(tokenized_datasets)
        example = train[train['kfold']==i].reset_index(drop=True)
        # example = Dataset.from_pandas(example)
        kfold_examples.append(example)
    data_collator = DataCollatorForTokenClassification(tokenizer)
    return kfold_tokenized_datasets, N_LABELS, data_collator, kfold_examples, tokenizer