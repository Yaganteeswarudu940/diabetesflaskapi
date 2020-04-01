import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle as p
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
df=pd.read_csv("C:/example files/files/diabetesIndia.csv")
x = df.iloc[:,0:8]
y = df.iloc[:,8]
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0)
weight = {
    1:1.05,
    0:1
    }
cols = df.columns.tolist()
features = cols.copy()
features.remove('Outcome')
features.remove('DiabetesPedigreeFunction')
features.remove('Pregnancies')
print(features)
log_model = LogisticRegression(class_weight = weight,max_iter = 4000)
log_model.fit(df[features], df['Outcome'])
p.dump(log_model, open('final_risk.pickle', 'wb'))

