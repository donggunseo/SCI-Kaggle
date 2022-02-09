'''
이 파일은 link evidnece를 설명하기 위한 파일입니다.
(by jspirit01)

link_evidence 함수는 인접해있는 evidence class를 서로 이어붙임으로써 false negative를 줄이고자 하는 후처리 작업입니다.
'''

from transformers import AutoTokenizer
from torch.utils.data import Dataset
import torch
import pandas as pd
from tqdm import tqdm

# 기본적으로 link evidence는 sumission할 df에 대해서 수행하지만, 이 파일에선 예제 설명을 위해 train.csv 파일을 가공하여 설명합니다.
train_df = pd.read_csv('input/feedback-prize-2021/train.csv')
train_df.rename(columns={'discourse_type':'class'}, inplace=True)
df = train_df[['id', 'class', 'predictionstring']]             # 이처럼, submission df는 3가지 열로 구성됩니다.


def link_evidence(oof):
    if not len(oof):
        return oof
  
    def jn(pst, start, end):
        return " ".join([str(x) for x in pst[start:end]])
  
    thresh = 1                      # 첫번째 threshold
    idu = oof['id'].unique()        # array(['423A1CA112E2', 'A8445CABFECE', ..., '4C471936CD75'], dtype=object)
    eoof = oof[oof['class'] == "Evidence"]
    neoof = oof[oof['class'] != "Evidence"]
    eoof.index = eoof[['id', 'class']]                                  # idx를 (id, class)로 재설정
    for thresh2 in range(26, 27, 1):                                    # 두번째 threshold (return 할거면서 왜 for문을 썼는지 모르겠음)
        retval = []
        idv = '423A1CA112E2'
        # for idv in tqdm(idu, desc='link_evidence', leave=False):      # 모든 id에 대하여 for loop
        for c in ['Evidence']:                                          # "Evidence에 대해서만 연결(link) 수행하겠다"
            q = eoof[(eoof['id'] == idv)]                               # class가 evidence인 행 중에 id가 idv인 행만 가져옴
            if len(q) == 0:
                continue
            pst = []
            for r in q.itertuples():
                pst = [*pst, -1,  *[int(x) for x in r.predictionstring.split()]]        # same id with evidence의 predictionstring을 구분자를 사이로 두고 하나의 리스트로 모은다. (구분자는 -1)
            
            # 
            # 인접한 class끼리 연결시키는 코드 -----------------------------------------------------------
            # 동작하는 방식은, 
            start = 1
            end = 1
            for i in range(2, len(pst)):
                cur = pst[i]
                end = i
                if  ((cur == -1) and ((pst[i + 1] > pst[end - 1] + thresh) or (pst[i + 1] - pst[start] > thresh2))):
                    # [조건1] cur == -1 : evidnce의 끝에 왔음 
                    #   &&
                    # [조건2]
                    # pst[i+1] > pst[end - 1] + thresh : 다음 evidence의 start word id가 현재 evidnece의 end word id (+threshold(1)) 보다 크다면 분리한다. 
                    #                                      (즉, 인접하지 않고 멀리 떨어져 있다면 분리한다.)
                    #   OR
                    # pst[i+1] - pst[start] > thresh2 : 다음 evidence의 start word id와 현재 evidence의 start word id의 차이가 threshold2(26)를 넘으면 분리한다.
                    #                                   (이 조건은, 비록 두 evidence가 인접해있더라도 분리시킬 수 있는 장치이다.)
                    retval.append((idv, c, jn(pst, start, end)))        # start word id~ end word id까지를 하나의 evidence row로 만들어 retval에 추가함
                    start = i + 1
            v = (idv, c, jn(pst, start, end + 1))       # for문에서 처리되지 못한 나머지 evidence 행을 마저 처리해줌.
            retval.append(v)
            # --------------------------------------------------------------------------------------
        roof = pd.DataFrame(retval, columns=['id', 'class', 'predictionstring'])
        roof = roof.merge(neoof, how='outer')
        return roof
    
submission = link_evidence(df)

