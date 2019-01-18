# Cat Boost Model 
# Notes: 
# 1) Catboost handles categorical values automatically. we just need to convert them into numeric using label encoder.
# 2) Catboost handles missing values automatically. However, to perform Standard scale before fitting the model 
# we need to fill the missing values to avoid NaN error. 
# 3) Stratified cross validation should be used when there are two classes(binary format).
# 4) Kfold cross validation should be used when there are more than two classes.
 

# Importing the Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Importing  the dataset
train = pd.read_csv("exercise_02_train.csv")
test = pd.read_csv("exercise_02_test.csv")
train.shape, test.shape

# Plot the distrinution of target variable
train["y"].astype("int").plot.hist()
# This is unbalanced dataset because target variable(1) is on minority side

# Combine the train and Test dataset for analysis and cleaning purpose
comb = pd.concat([train,test], axis = 0, ignore_index=True)
comb.shape
comb.dtypes.value_counts()


################## Examine and Imputing the missing value ###################

def find_missing_value(data):
    mis_value = data.isnull().sum()
    mis_value_percent = mis_value/len(data)*100
    mis_value_table = pd.concat([mis_value,mis_value_percent], axis=1)
    mis_value_table_rename = mis_value_table.rename(columns = {0:"Missing value", 1:"% of Total value"})
    mis_value_table_rename = mis_value_table_rename[mis_value_table_rename.iloc[:,1]!=0].sort_values("% of Total value", ascending = False).round(1)
    print("There are total "+ str(mis_value_table_rename.shape[0]) + " variables that have missing values")
    return mis_value_table_rename

find_missing_value(comb.drop(columns="y"))


# Fill the missing values in continous variables
from sklearn.preprocessing import Imputer
comb_label = comb["y"]
comb = comb.drop(columns="y")
imputer = Imputer(missing_values = 'NaN', strategy = 'median', axis = 0)
for i in comb.select_dtypes(["float64","int64"]):
    if comb[i].isnull().sum()>0:
        comb[i] = comb[i].values
        imputer = imputer.fit(comb[i].values.reshape(-1,1))
        comb[i] = imputer.transform(comb[i].values.reshape(-1,1))
print("Remaining missing values after imputing: ", find_missing_value(comb))


########################## Treatment on categorical features ########################

# 1] check the categorical variables for anamolies in combined dataset
comb.select_dtypes('object').apply(pd.Series.nunique, axis = 0)
# There are 5 categorical variables in the combined dataset

# 2] Find Number of counts in each categorical variables

for i in comb.select_dtypes("object"):
    print(comb[i].value_counts())
    print()
    # Classes in x35 can be club further wherever possible  e.g. fri & friday becomes one class together

# x35
print ('Original Categories:')
print (comb['x35'].value_counts())
print ('\nModified Categories:')
comb["x35"] = comb["x35"].replace({"fri":'friday',
                                     "thur":"thurday",
                                     "wed":"wednesday"})
print (comb["x35"].value_counts())

# 3) convert categorical features into int before construct Dataset
comb.dtypes.value_counts()
from sklearn.preprocessing import LabelEncoder

# label encoder
# CatBoost handles categorical features automatically, we just need to convert classes into int using labelencoder
le = LabelEncoder()
count=0
for i in comb:
    if comb[i].dtypes=="object":  
        if len(list(comb[i].unique())) > 0:
            comb[i] = le.fit(comb[i].astype(str)).transform(comb[i].astype(str))
            count += 1
            
print('No. of columns were encoded: ', count)            


#################### Treatment on continous variable #########################
for i in comb.select_dtypes("float64"):
    print(comb[i].describe())

# convert the -ve $ into absolute $
comb["x41"] = abs(comb["x41"])

# Split the comb into train and test dataset
comb = pd.merge(pd.DataFrame(comb), pd.DataFrame(comb_label), how="left", right_index = True, left_index = True)
train = comb[comb["y"].notnull()]
test = comb[comb["y"].isnull()]
test = test.drop(columns="y")
comb.shape, train.shape, test.shape


############################### CatBoost model model ################################

# Split the train dataset 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train.iloc[:,:-1], train.iloc[:,-1], train_size= 0.7, stratify = train.y, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# Fit the Catboost model
from catboost import CatBoostClassifier
categorical_features_name = ["x34", "x35", "x45", "x68", "x93"]
cat_dims = [X_train.columns.get_loc(i) for i in categorical_features_name[:-1]] 
clf = CatBoostClassifier(iterations = 100, 
                         learning_rate = 0.1,
                         depth = 6,
                         l2_leaf_reg = 3,
                         loss_function = "Logloss",         # for 2 class logloss and more than 2 classes Multiclass
                         border_count = 32)


clf.fit(X_train, y_train, 
        cat_dims, 
        early_stopping_rounds = 100, 
        eval_set= [(X_train, y_train)],
        verbose_eval=True)   

pred= clf.predict(X_test)
prob = clf.predict_proba(X_test)
clf.feature_importances_

# Cross Validation
from sklearn.model_selection import cross_val_predict, StratifiedKFold
cv = StratifiedKFold(n_splits=3, random_state=0)
pred = cross_val_predict(estimator = clf, X = X_train, y = y_train, cv = cv)
prob = cross_val_predict(estimator = clf, X = X_train, y = y_train, cv = cv, method = "predict_proba")

# Analyze the performance of Catboost model using confusion matrix and AUC
from sklearn.metrics import confusion_matrix, average_precision_score, auc, roc_curve
cm =confusion_matrix(y_test, pred)

average_precision = average_precision_score(y_test, prob[:,1])
print("Average precision Recall score: ",(average_precision))

fpr, tpr,threshold_ = roc_curve(y_test, prob[:,1])
roc_auc = auc(fpr, tpr)                
print("AUC score: ",(roc_auc))


# Tunned the model using Gridsearch
from sklearn.model_selection import GridSearchCV

grid_params = {'depth':[3,1,2,6,4,5,7,8,9,10],
          'iterations':[500,1000],
          'learning_rate':[0.03,0.1,0.2], 
          'l2_leaf_reg':[3,1,5],
          'border_count':[32,20],
          'thread_count':[4]}

grid_search = GridSearchCV(estimator = clf, 
                           param_grid = grid_params, 
                           scoring = 'accuracy', 
                           cv = cv, 
                           n_jobs = -1,
                           verbose = 3)

grid_search = grid_search.fit(X_train, y_train, early_stopping_rounds = 100)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_


# Save the model
clf.save_model("Catboost", format = "cbm")

# Load the model
clf = CatBoostClassifier()
clf.load_model("Catboost")
