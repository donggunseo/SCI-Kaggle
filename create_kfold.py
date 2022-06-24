import pandas as pd

from sklearn.model_selection import KFold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

def create_kfold(df, k=5):
    dfx = pd.get_dummies(df, columns=["discourse_type"]).groupby(["id"], as_index=False).sum()
    ## 출력 : ['id', 'discourse_type_Claim', 'discourse_type_Concluding Statement', 'discourse_type_Counterclaim', 'discourse_type_Evidence', 'discourse_type_Lead', 'discourse_type_Position','discourse_type_Rebuttal']
    cols = [c for c in dfx.columns if c.startswith("discourse_type_") or c == "id" and c != "discourse_type_num"]
    dfx = dfx[cols]

    mskf = MultilabelStratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    labels = [c for c in dfx.columns if c != "id"]
    dfx_labels = dfx[labels]
    dfx["kfold"] = -1

    for fold, (trn_, val_) in enumerate(mskf.split(dfx, dfx_labels)):
        dfx.loc[val_, "kfold"] = fold
    
    df = df.merge(dfx[["id", "kfold"]], on="id", how="left")
    return dfx[["id", "kfold"]], df