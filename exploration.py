import pandas as pd
import os
import ppscore as pps
import seaborn as sns
from nameparser import HumanName

data_folder = os.getcwd() + '/data/input/'

train_df = pd.read_csv(data_folder + 'train.csv')

train_df.describe()
matrix_df = pps.matrix(train_df[['Survived','Pclass','Sex','SibSp','Parch','Age','Fare','Embarked','Ticket','Cabin']])[['x', 'y', 'ppscore']].pivot(columns='x', index='y', values='ppscore')
sns.heatmap(matrix_df, vmin=0, vmax=1, cmap="Blues", linewidths=0.5, annot=True)
print()
#only fare, sex and ticket are good uni-predictors
#can anything useful be extracted from name?
#mr. miss. mrs. master. don. dr. rev. mme. major
#extract first letter of cabin

train_df["title"] = train_df["Name"].apply(lambda x: HumanName(x).title)
train_df["first"] = train_df["Name"].apply(lambda x: HumanName(x).first)
train_df["nickname"] = train_df["Name"].apply(lambda x: HumanName(x).nickname)
train_df.loc[train_df["title"] == '','title'] = train_df["first"]
train_df['has_nickname'] = 1
train_df.loc[train_df["nickname"] == '','has_nickname'] = 0

train_df.Cabin = train_df.Cabin.str[0]

matrix_df = pps.matrix(train_df[['Survived','title','has_nickname','Cabin']])[['x', 'y', 'ppscore']].pivot(columns='x', index='y', values='ppscore')
sns.heatmap(matrix_df, vmin=0, vmax=1, cmap="Blues", linewidths=0.5, annot=True)
