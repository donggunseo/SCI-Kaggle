# SCI-Kaggle
Competition Link : https://www.kaggle.com/c/feedback-prize-2021/overview

## requirement
`$pip install -r requirements.txt`

## Preprocess 
`$ python preprocess_pd.py`

## Oversampling
Download <u>glove.6B.zip</u> from https://nlp.stanford.edu/projects/glove/

Unzip and Place it as ../glove/glove.6B/glove.6B.50d.txt

`$ python oversampling.py`
## Training
`$ python train.py`

## inference
Run Submission.ipynb