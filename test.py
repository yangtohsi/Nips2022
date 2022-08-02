from datautil import DatasetLoader
from features import FeatureBinarizer
from ssrl import submodular_rule_list_learner
import numpy as np
import pandas as pd


name = 'WDBC'

df = DatasetLoader(name).dataframe

column_list = list(df.columns)
column_list.remove('label')

binarizer = FeatureBinarizer(numThresh=3, negations=False, threshStr=False)
df_X = binarizer.fit_transform(df[column_list])

feature_columns = df_X.columns
df_ = pd.concat((df_X,df['label']),axis=1).astype('str')
rule_learner = submodular_rule_list_learner(lambda_1=7,
                                        distorted_step=10,cc=10,use_multi_pool=False)

rule_learner.train(dataset=df_,
                   label_column='label',
                   feature_columns=feature_columns)

pred_test = rule_learner.predict(df_[feature_columns])
acc = np.sum(1.0*(pred_test.values==df_[['label']].values))/df_.shape[0]
print('acc='+str(acc))
