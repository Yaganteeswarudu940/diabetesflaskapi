#Importing required values

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve,GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve,auc
from sklearn.preprocessing import binarize
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
from sklearn.datasets import make_classification
matplotlib.use('Agg')

#Reading csv file

df=pd.read_csv("C:/example files/files/diabetesIndia.csv")
df.head(5)


#Declaring required variables for stroring scores

names=[]
npvalues=[]
PPV=[]
sensiti=[]
specifi=[]
perfor=[]
prevlance=[]
fnr=[]
fpr=[]
tnr=[]
tpr=[]
aucperf=[]


# Function to calculate performace measures
def NPV(tn, fp, fn, tp):
    PPV = round( tp / (tp+fp),4 )*100
    NPValues = round( tn / (tn+fn),4 )*100
    Population = tn+fn+tp+fp
    Accuracy=round( (tp+tn) / Population,4)*100
    sensitivity  = round(tp / (tp+fn),4)*100
    specificity  = round(tn / (tn+fp),4)*100
    Prevalence = round(tp / Population,2)*100
    FPR        = round( fp / (tn+fp),4 )*100
    FNR        = round( fn / (tp+fn),4 )*100
    TNR        = round( tn / (tn+fp),4 )*100
    TPR        = round(tp/ (tp + fn),4)*100
    return PPV,NPValues,Accuracy,sensitivity,specificity,Prevalence,FPR,FNR,TNR,TPR


def calculateperformance(y_test, y_pred,name):
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    ppv,npv,acc,s,sf,prev,f1,f2,f3,f4=NPV(tn, fp, fn, tp)
    npvalues.append(npv)
    PPV.append(ppv)
    perfor.append(acc)
    sensiti.append(s)
    specifi.append(sf)
    names.append(name)
    prevlance.append(prev)
    fnr.append(f2)
    fpr.append(f1)
    tnr.append(f3)
    tpr.append(f4)


# Function to create AUCCurve
def AUCCurve(model,name):
    # predict probabilities
    lr_probs = model.predict_proba(x_test)
    # keep probabilities for the positive outcome only
    lr_probs = lr_probs[:, 1]
    # calculate scores
    lr_auc = roc_auc_score(y_test, lr_probs)
    print('Logistic: ROC AUC=%.3f' % (lr_auc))
    aucperf.append(round(lr_auc,4)*100)
    # calculate roc curves
    #ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
    # plot the roc curve for the model
    #pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    plt.plot(lr_fpr, lr_tpr, marker='.', label=name)
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.legend()
    # show the plot
    plt.show()

#Print sum of null values
print(df.isnull().sum())


# Dividing data tnto trainset and test set
x = df.iloc[:,0:8]
y = df.iloc[:,8]
#x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)


#Selecting Best Features

bestfeatures = SelectKBest(score_func=chi2, k=8)
fit = bestfeatures.fit(x,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(x.columns) 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Features','Score']  
print(featureScores.nlargest(8,'Score'))


#Random forest classifier

model = RandomForestClassifier()
model.fit(x,y)
print(model.feature_importances_) 
feat_importances = pd.Series(model.feature_importances_, index=x.columns)
feat_importances.nlargest(8).plot(kind='barh')
plt.show()


# to find the correlation between features
correlationmat=df.corr()
top_corr_features=correlationmat.index
plt.figure(figsize=(10,10))
df12=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap='RdYlGn')


# Feature Extraction with PCA
import numpy
from pandas import read_csv
from sklearn.decomposition import PCA
# feature extraction
pca = PCA(n_components=3)
fit = pca.fit(x)
# summarize components
#print("Explained Variance: %s" % fit.explained_variance_ratio_)
print(fit.components_)


#Display chart based on outcome

df.groupby('Outcome').size()



df.groupby('Outcome').hist(figsize=(10, 10))


#Data Preprocessing
# BP values whose input is zero

print("Total : ", df[df.BloodPressure == 0].shape[0])
print(df[df.BloodPressure == 0].groupby('Outcome')['Age'].count())


# Glucose values whose input is zero

print("Total : ", df[df.Glucose == 0].shape[0])
print(df[df.Glucose == 0].groupby('Outcome')['Age'].count())


#Skin thickness whose input value is zero
print("Total : ", df[df.SkinThickness == 0].shape[0])
print(df[df.SkinThickness == 0].groupby('Outcome')['Age'].count())


# Records with BMI zero

print("Total : ", df[df.BMI == 0].shape[0])
print(df[df.BMI == 0].groupby('Outcome')['Age'].count())


#Records with insulin zero

print("Total : ", df[df.Insulin == 0].shape[0])
print(df[df.Insulin == 0].groupby('Outcome')['Age'].count())


# Preprocessing records
diabetes_mod = df[(df.BloodPressure != 0) & (df.BMI != 0) & (df.Glucose != 0)]
print(diabetes_mod.shape)


# Seperating input and output

x = diabetes_mod.iloc[:,0:8]
y = diabetes_mod.iloc[:,8]
#train , test = train_test_split(x,y,test_size = 0.2,random_state = 123)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=123)


# Naive bayes classifier algorithm

from sklearn.naive_bayes import GaussianNB
naivebayesclassifier = GaussianNB()
naivebayesclassifier.fit(x_train, y_train)
naivebayesclassifier.class_prior_
y_pred=naivebayesclassifier.predict(x_test)
calculateperformance(y_test, y_pred,"NB")
AUCCurve(naivebayesclassifier,"NB")
#accuracyrf = round(accuracy_score(y_pred, y_test), 5)
#accuracyrf




#GPC algorithm

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
kernel = 1.0 * RBF([1.0])
gpc_rbf_isotropic = GaussianProcessClassifier(kernel=kernel).fit(x_train, y_train)
y_pred=gpc_rbf_isotropic.predict(x_test)
calculateperformance(y_test, y_pred,"GPC")
AUCCurve(gpc_rbf_isotropic,"GPC")


#from sklearn.naive_bayes import GaussianNB
#naivebayesclassifier = GaussianNB(priors=[0.7041502, 0.2958498])
#naivebayesclassifier.fit(x_train, y_train)
#y_pred=naivebayesclassifier.predict(x_test)
#accuracyrf = round(accuracy_score(y_pred, y_test), 5)
#accuracyrf


#RF
rmfr = RandomForestClassifier()
rmfr.fit(x_train, y_train)
y_pred = rmfr.predict(x_test)
calculateperformance(y_test, y_pred,"RFC")
AUCCurve(rmfr,"RFC")
#accuracyrf = round(accuracy_score(y_pred, y_test), 5)
#models.append("RFC")
#scores.append(accuracyrf)
#accuracyrf



#Random forest parameter tuning
#n_estimators represents the number of trees in the forest. Usually the higher the number of trees the better to learn the data.
#However, adding a lot of trees can slow down the training process considerably, therefore we do a parameter search to find the
#sweet spot.
# n_estimators = number of tress
n_estimators = [1, 2, 4, 8, 16, 32, 64, 100, 200]
train_results = []
test_results = []
for estimator in n_estimators:
    rf = RandomForestClassifier(n_estimators=estimator, n_jobs=-1)
    rf.fit(x_train, y_train)
    train_pred = rf.predict(x_train)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    train_results.append(roc_auc)
    y_pred = rf.predict(x_test)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    test_results.append(roc_auc)
from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(n_estimators, train_results, 'b', label="Diabetes Train set AUC")
line2, = plt.plot(n_estimators, test_results, "r", label="Diabetes Test set AUC")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('AUC score')
plt.xlabel('n_estimators')
plt.show()


rmfr1 = RandomForestClassifier(n_estimators=50)
rmfr1.fit(x_train, y_train)
y_pred = rmfr1.predict(x_test)
#accuracyrf = round(accuracy_score(y_pred, y_test), 5)
#models.append("RFC_estimators")
#scores.append(accuracyrf)
calculateperformance(y_test, y_pred,"RFestimators50")
AUCCurve(rmfr1,"RFEstimator50")
#accuracyrf


#max_depth tuning in random forest
#max_depth represents the depth of each tree in the forest. The deeper the tree, the more splits it has and it captures more
#information about the data. We fit each decision tree with depths ranging from 1 to 32 and plot the training and test errors.

max_depths = np.linspace(1, 32, 32, endpoint=True)
train_results = []
test_results = []
for max_depth in max_depths:
    rf = RandomForestClassifier(max_depth=max_depth, n_jobs=-1)
    rf.fit(x_train, y_train)
    train_pred = rf.predict(x_train)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    train_results.append(roc_auc)
    y_pred = rf.predict(x_test)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    test_results.append(roc_auc)
from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(max_depths, train_results, 'b', label="Diabetes Train set AUC")
line2, = plt.plot(max_depths, test_results, 'r', label="Diabetes Test set AUC")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('AUC score')
plt.xlabel('Tree depth')
plt.show()


rmfr2 = RandomForestClassifier(max_depth=15,n_jobs=-1)
rmfr2.fit(x_train, y_train)
y_pred = rmfr2.predict(x_test)
#accuracyrf = round(accuracy_score(y_pred, y_test), 5)
#models.append("RFC_max_depth")
#scores.append(accuracyrf)
calculateperformance(y_test, y_pred,"RFmaxdepth15")
AUCCurve(rmfr2,"RFmaxdepth15")
#accuracyrf


#min_samples_split
#min_samples_split represents the minimum number of samples required to split an internal node. This can vary between
#considering at least one sample at each node to considering all of the samples at each node. When we increase this parameter,
#each tree in the forest becomes more constrained as it has to consider more samples at each node.
#Here we will vary the parameter from 10% to 100% of the samples

min_samples_splits = np.linspace(0.1, 1.0, 10, endpoint=True)
train_results = []
test_results = []
for min_samples_split in min_samples_splits:
    rf = RandomForestClassifier(min_samples_split=min_samples_split)
    rf.fit(x_train, y_train)
    train_pred = rf.predict(x_train)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    train_results.append(roc_auc)
    y_pred = rf.predict(x_test)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    test_results.append(roc_auc)
from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(min_samples_splits, train_results, 'b', label="Diabetes Train set AUC")
line2, = plt.plot(min_samples_splits, test_results, 'r', label="Diabetes Test set AUC")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('AUC score')
plt.xlabel('min samples split')
plt.show()


rmfr3 = RandomForestClassifier(min_samples_split=0.1)
rmfr3.fit(x_train, y_train)
y_pred = rmfr3.predict(x_test)
#accuracyrf = round(accuracy_score(y_pred, y_test), 5)
#models.append("RFC_min_samples_split")
#scores.append(accuracyrf)
calculateperformance(y_test, y_pred,"RFminsamples")
AUCCurve(rmfr3,"RFminsamples")
#accuracyrf



#min_samples_leaf
#min_samples_leaf is The minimum number of samples required to be at a leaf node. This parameter is similar 
#to min_samples_splits, however, this describe the minimum number of samples of samples at the leafs, the base of the tree.
min_samples_leafs = np.linspace(0.1, 0.5, 5, endpoint=True)
train_results = []
test_results = []
for min_samples_leaf in min_samples_leafs:
    rf = RandomForestClassifier(min_samples_leaf=min_samples_leaf)
    rf.fit(x_train, y_train)
    train_pred = rf.predict(x_train)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    train_results.append(roc_auc)
    y_pred = rf.predict(x_test)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    test_results.append(roc_auc)
from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(min_samples_leafs, train_results, 'b', label="Diabetes Train set AUC")
line2, = plt.plot(min_samples_leafs, test_results, 'r', label="Diabetes Test set AUC")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('AUC score')
plt.xlabel('min samples leaf')
plt.show()



rmfr4 = RandomForestClassifier(min_samples_leaf=0.1)
rmfr4.fit(x_train, y_train)
y_pred = rmfr4.predict(x_test)
#accuracyrf = round(accuracy_score(y_pred, y_test), 5)
#models.append("RFC_min_samples_leaf")
#scores.append(accuracyrf)
#accuracyrf
calculateperformance(y_test, y_pred,"RFminsamplesleaf")
AUCCurve(rmfr4,"RFminsamplesleaf")




max_features = list(range(1,x.shape[1]))
train_results = []
test_results = []
for max_feature in max_features:
    rf = RandomForestClassifier(max_features=max_feature)
    rf.fit(x_train, y_train)
    train_pred = rf.predict(x_train)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    train_results.append(roc_auc)
    y_pred = rf.predict(x_test)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    test_results.append(roc_auc)
from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(max_features, train_results, 'b', label="Diabetes Train set AUC")
line2, = plt.plot(max_features, test_results, 'r', label="Diabetes Test set AUC")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('AUC score')
plt.xlabel('max features')
plt.show()



rmfr5 = RandomForestClassifier(max_features=5)
rmfr5.fit(x_train, y_train)
y_pred = rmfr5.predict(x_test)
#accuracyrf = round(accuracy_score(y_pred, y_test), 5)
#models.append("RFC_min_features")
#scores.append(accuracyrf)
#accuracyrf
calculateperformance(y_test, y_pred,"RFminfeatures")
AUCCurve(rmfr5,"RFminfaetures5")



rf = RandomForestClassifier(n_estimators=120, max_features=7)
rf.fit(x_train,y_train)
pred = rf.predict(x_test)
#accuracy = round(accuracy_score(y_test, pred),5)
#accuracy
#print('\n')
#print(confusion_matrix(y_test,pred))
#print('\n')
#print(classification_report(y_test,pred))



from sklearn.tree import DecisionTreeClassifier
# Create Decision Tree classifer object
clf = DecisionTreeClassifier(random_state=42)
# Train Decision Tree Classifer
clf = clf.fit(x_train,y_train)
#Predict the response for test dataset
y_pred = clf.predict(x_test)
calculateperformance(y_test, y_pred,"DT")
AUCCurve(clf,"DT")
#from sklearn.metrics import accuracy_score
#models.append("DT")
#accuracydt=accuracy_score(y_test,y_pred)
#scores.append(accuracydt)
#print("Accuracy: ",accuracydt)



# Deciosn tree tuning change the criterion to entropy
clf1 = DecisionTreeClassifier(criterion = 'entropy')
clf1 = clf1.fit(x_train,y_train)
y_pred = clf1.predict(x_test)
#models.append("DTEntropy")
#accuracydt=accuracy_score(y_test,y_pred)
#scores.append(accuracydt)
#print("Accuracy: ",accuracydt)
calculateperformance(y_test, y_pred,"DTEntropy")
AUCCurve(clf1,"DTEntropy")



max_depths = np.linspace(1, 32, 32, endpoint=True)
train_results = []
test_results = []
for max_depth in max_depths:
    dt = DecisionTreeClassifier(max_depth=max_depth)
    dt.fit(x_train, y_train)
    train_pred = dt.predict(x_train)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    train_results.append(roc_auc)
    y_pred = rf.predict(x_test)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    test_results.append(roc_auc)
from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(max_depths, train_results, 'b', label="Diabetes Train set AUC")
line2, = plt.plot(max_depths, test_results, 'r', label="Diabetes Test set AUC")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('AUC score')
plt.xlabel('Tree depth')
plt.show()



# Deciosn tree tuning change the criterion to entropy
clf2 = DecisionTreeClassifier(max_depth=2)
clf2 = clf.fit(x_train,y_train)
y_pred = clf2.predict(x_test)
#models.append("DT_max_depth")
#accuracydt=accuracy_score(y_test,y_pred)
#scores.append(accuracydt)
#print("Accuracy: ",accuracydt)
calculateperformance(y_test, y_pred,"DTmaxdepth")
AUCCurve(clf2,"DTmaxdepth")



min_samples_splits = np.linspace(0.1, 1.0, 10, endpoint=True)
train_results = []
test_results = []
for min_samples_split in min_samples_splits:
    dt = DecisionTreeClassifier(min_samples_split=min_samples_split)
    dt.fit(x_train, y_train)
    train_pred = dt.predict(x_train)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    train_results.append(roc_auc)
    y_pred = rf.predict(x_test)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    test_results.append(roc_auc)
from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(min_samples_splits, train_results, 'b', label="Diabetes Train set AUC")
line2, = plt.plot(min_samples_splits, test_results, 'r', label="Diabetes Test set AUC")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('AUC score')
plt.xlabel('min samples split')
plt.show()


# Deciosn tree tuning change the criterion to entropy
clf3 = DecisionTreeClassifier(min_samples_split=0.2)
clf3 = clf.fit(x_train,y_train)
y_pred = clf3.predict(x_test)
#models.append("DT_min_samples_split")
#accuracydt=accuracy_score(y_test,y_pred)
#scores.append(accuracydt)
#print("Accuracy: ",accuracydt)
calculateperformance(y_test, y_pred,"DTMinsamples")
AUCCurve(clf3,"DTMinsamples")


#XGB algorithm

from xgboost import XGBClassifier
xgb = XGBClassifier()
xgb.fit(x_train, y_train)
y_pred = xgb.predict(x_test)
predictions = [round(value) for value in y_pred]
calculateperformance(y_test, y_pred,"XGB")
AUCCurve(xgb,"XGB")
accuracy = round(accuracy_score(y_test, predictions),5)
#models.append("XGBC")
#scores.append(accuracy)
accuracy

#LR algorithm

from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression(max_iter = 4000)
#** Train/fit lm on the training data.**
logmodel.fit(x_train,y_train)
predictions = logmodel.predict(x_test)
calculateperformance(y_test, predictions,"LR")
AUCCurve(logmodel,"LR")
#accuracy = round(accuracy_score(y_test, predictions),5)
#models.append("LR")
#scores.append(accuracy)
#accuracy


#Kernel
#kernel parameters selects the type of hyperplane used to separate the data. Using ‘linear’ will use a linear hyperplane 
#(a line in the case of 2D data). ‘rbf’ and ‘poly’ uses a non linear hyper-plane
from sklearn import svm
sv = svm.SVC(kernel='linear',probability=True,C=1,random_state=42)
sv.fit(x_train,y_train)
pred = sv.predict(x_test)
calculateperformance(y_test, pred,"SVM")
AUCCurve(sv,"SVM")
accuracy = round(accuracy_score(y_test, pred),5)
#models.append("SVM")
#scores.append(accuracy)
accuracy


# parameter tunig in svc by changing kernal
from sklearn import svm
sv = svm.SVC(kernel='rbf')
sv.fit(x_train,y_train)
pred = sv.predict(x_test)
accuracy = round(accuracy_score(y_test, pred),5)
#models.append("SVMrbf")
#scores.append(accuracy)
accuracy



#from sklearn import svm
#sv = svm.SVC(kernel='poly')
#sv.fit(x_train,y_train)
#pred = sv.predict(x_test)
#accuracy = round(accuracy_score(y_test, pred),5)
#models.append("SVMpoly")
#scores.append(accuracy)
#accuracy


# parameter tunig in svc by changing gamma

#gamma
#gamma is a parameter for non linear hyperplanes. The higher the gamma value it tries to exactly fit the training data set
#from sklearn import svm
#sv = svm.SVC(kernel='rbf',gamma=0.5)
#sv.fit(x_train,y_train)
#pred = sv.predict(x_test)
#accuracy = round(accuracy_score(y_test, pred),5)
#accuracy


#C
#C is the penalty parameter of the error term. It controls the trade off between smooth decision boundary and classifying the 
#training points correctly.
#from sklearn import svm
#sv = svm.SVC(kernel='rbf',C=0.1)
#sv.fit(x_train,y_train)
#pred = sv.predict(x_test)
#accuracy = round(accuracy_score(y_test, pred),5)
#accuracy


#from sklearn import svm
#sv = svm.SVC(kernel='rbf',C=1,probability=True)
#sv.fit(x_train,y_train)
#pred = sv.predict(x_test)
#accuracy = round(accuracy_score(y_test, pred),5)
#accuracy



from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(x_train)
x_train_transformed = scaler.transform(x_train)
clf = svm.SVC(C=1).fit(x_train_transformed, y_train)
x_test_transformed = scaler.transform(x_test)
clf.score(x_test_transformed, y_test)



gbi = GradientBoostingClassifier(learning_rate=0.05,max_depth=3,max_features=0.5)
gbi.fit(x_train,y_train)



#STORING THE PREDICTION
yprediction = gbi.predict_proba(x_test)[:,1]
from sklearn.metrics import classification_report, confusion_matrix
cm = confusion_matrix(y_test, yprediction.round())
#calculateperformance(y_test, yprediction,"GBC")
#round(roc_auc_score(y_test,yprediction),5)
#models.append("gbi")
#scores.append(roc_auc_score(y_test,yprediction))
ppv,npv,acc,s,sf,prev,f1,f2,f3,f4=NPV(cm[0,0], cm[0,1], cm[1,0], cm[1,1])
npvalues.append(npv)
PPV.append(ppv)
perfor.append(acc)
sensiti.append(s)
specifi.append(sf)
names.append("GBC")
prevlance.append(prev)
fnr.append(f2)
fpr.append(f1)
tnr.append(f3)
tpr.append(f4)
aucperf.append(80.9)


#LDA algorithm

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()
lda.fit(x_train,y_train)
pred_lda = lda.predict(x_test)
calculateperformance(y_test, pred_lda,"LDA")
AUCCurve(lda,"LDA")
#models.append("LDA")
#scores.append(accuracy_score(y_test,pred_lda))
#print("Accuracy: ",accuracy_score(y_test,pred_lda))

#QDA algorithm

qda = QuadraticDiscriminantAnalysis()
qda.fit(x_train,y_train)
pred_qda = qda.predict(x_test)
calculateperformance(y_test, pred_qda,"QDA")
AUCCurve(qda,"QDA")


from sklearn.metrics import roc_auc_score
auc = roc_auc_score(y_test, pred_qda)
print('AUC: %.3f' % auc)


from sklearn import model_selection
numsplits=[2,4,5,10]
for i in numsplits:
    kfold = model_selection.KFold(n_splits=i, random_state=100,shuffle=True)
    model_kfold = QuadraticDiscriminantAnalysis()
    results_kfold = model_selection.cross_val_score(model_kfold, x_train, y_train, cv=kfold)
    print("Accuracy: %.2f%%" % (results_kfold.mean()*100.0)) 


#KNN
    
#Standardize the Variables
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
scaler = StandardScaler()
scaler.fit(df.drop('Outcome',axis=1))
scaled_features = scaler.transform(df.drop('Outcome',axis=1))
df_feat = pd.DataFrame(scaled_features,columns=df.columns[:-1])
# Using KNN
#from sklearn.model_selection import train_test_split
#x_train, x_test, y_train, y_test = train_test_split(scaled_features,df['Outcome'],
                                                   # test_size=0.25,random_state=200)
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train,y_train)
pred = knn.predict(x_test)
calculateperformance(y_test, pred,"KNN")
AUCCurve(knn,"KNN")
#accuracy = round(accuracy_score(y_test, pred),5)
#models.append("KNN")
#scores.append(accuracy)
#accuracy


#KNN parameter tuning
# NOW WITH K=19
knn = KNeighborsClassifier(n_neighbors=16)
knn.fit(x_train,y_train)
pred = knn.predict(x_test)
accuracy = round(accuracy_score(y_test, pred),5)
accuracy
#print('WITH K=19')
#print('\n')
#print(confusion_matrix(y_test,pred))
#print('\n')
#print(classification_report(y_test,pred))



#sns.pairplot(df, size=3, hue='Outcome', palette='husl',)
#plt.show()



train_score=pd.DataFrame({'Model':names , 'Accuracy': perfor , 'NPV' : npvalues , 'PPV' : PPV, 'Sensitivity' : sensiti ,'Specificity' : specifi, 'FNR' : fnr, 'FPR' :fpr,'TNR' :tnr,'TPR' :tpr,'AUC' :aucperf })



#train_score
train_score.sort_values("Accuracy", axis = 0, ascending = False, 
                 inplace = True, na_position ='last') 
train_score



#train_score1=train_score[train_score["Scores"]>0.73]



#axis = sns.barplot(x = 'Model', y = 'Accuracy', data = train_score)
#axis.set(xlabel='Classifier', ylabel='Accuracy')
#plt.figure(figsize=(15,11))
#for p in axis.patches:
    #height = p.get_height()
    #axis.text(p.get_x() + p.get_width()/2, height+0.005, '{:1.4f}'.format(height), ha="center") 
    #axis.set_xticklabels(axis.get_xticklabels(), rotation=40, ha="right")
    #plt.tight_layout()
#plt.show()



#from sklearn import tree
#plt.figure(figsize=(40,20))  # customize according to the size of your tree
#_ = tree.plot_tree(clf, feature_names = x.columns,filled=True, fontsize=20, rounded = True)
#plt.show()


# Load libraries
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn import metrics
svc=SVC(probability=True, kernel='linear')
abc =AdaBoostClassifier()
model = abc.fit(x_train, y_train)
y_pred = model.predict(x_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# Neual Network MLP classifier
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(random_state=42)
mlp.fit(x_train, y_train)
print("Accuracy on training set: {:.2f}".format(mlp.score(x_train, y_train)))
print("Accuracy on test set: {:.2f}".format(mlp.score(x_test, y_test)))


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.fit_transform(x_test)
mlp = MLPClassifier(random_state=0)
mlp.fit(x_train_scaled, y_train)
print("Accuracy on training set: {:.3f}".format(mlp.score(x_train_scaled, y_train)))
print("Accuracy on test set: {:.3f}".format(mlp.score(x_test_scaled, y_test)))


mlp = MLPClassifier(max_iter=1000, random_state=0)
mlp.fit(x_train_scaled, y_train)
print("Accuracy on training set: {:.3f}".format(
    mlp.score(x_train_scaled, y_train)))
print("Accuracy on test set: {:.3f}".format(mlp.score(x_test_scaled, y_test)))


#tensorflow
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense

import numpy

model = Sequential()

# 8 inputs, 12 outputs
# Rectifier activation layer
# https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
model.add(Dense(12, input_dim=8, activation='relu'))

# 12 inputs, 8 outputs
model.add(Dense(8, activation='relu'))

# ensure output is in range (0,1)
# Sigmoid activation layer
# https://en.wikipedia.org/wiki/Sigmoid_function
model.add(Dense(1, activation='sigmoid'))

# COMPILE MODEL

# logarithmic loss = binary cross entropy (https://en.wikipedia.org/wiki/Cross_entropy)
# gradient descent algorithm = adam (http://arxiv.org/abs/1412.6980)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# TRAINING

# epochs is the number of iterations
# batch size is the number of instances evaluated before weight update in network
model.fit(x_train, y_train, epochs=150, batch_size=10)

# EVALUATION (should be another dataset than we trained on)
scores = model.evaluate(x_test,y_test)

print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))




