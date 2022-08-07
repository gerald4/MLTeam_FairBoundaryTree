# This code is taken from https://github.com/kirtanp/MAMO-fair/blob/master/data_preprocessing.ipynb

import pandas as pd
import numpy as np


def one_hot_encode(df, cat_cols):
    for col in cat_cols:
        one_hot_enc = pd.get_dummies(df[col], prefix=col)
        df = df.join(one_hot_enc)
        df = df.drop([col], axis=1)
    return(df)

adult = pd.read_csv('adult-full.csv')

adult = adult.drop(['education', 'fnlwgt', 'capitalgain', 'capitalloss'], axis=1)
adult['class'] = adult['class'].apply(lambda x: 0 if x=='<=50K' else 1)

counts = adult['native-country'].value_counts()
replace = counts[counts <= 150].index
adult['native-country'] = adult['native-country'].replace(replace, 'other')

cat_cols_adult = ['workclass', 'marital-status', 'occupation', 'relationship', 'native-country']
adult = one_hot_encode(adult, cat_cols_adult)

adult['sex'] = adult['sex'].apply(lambda x: 1 if x=='Male' else 0)
adult['race'] = adult['race'].apply(lambda x: 1 if x=='White' else 0)

cols = adult.columns.tolist()


del cols[cols.index("sex")]
del cols[cols.index("race")]
del cols[cols.index("class")]

cols = cols + ["race", "sex", "class"]

adult[cols].to_csv("adult_processed.csv", index=False)


