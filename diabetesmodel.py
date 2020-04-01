import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve,GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.preprocessing import binarize
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import svm
import pickle as p
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
df=pd.read_csv("C:/example files/files/diabetesIndia.csv")
x = df.iloc[:,0:8]
y = df.iloc[:,8]
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0)
sv = svm.SVC(kernel='linear')
sv.fit(x_train,y_train)
pred = sv.predict(x_test)
p.dump(sv, open('final_diabetes.pickle', 'wb'))








