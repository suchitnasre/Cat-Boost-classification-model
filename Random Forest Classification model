# Classification model using Random Forest

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# Importing the Train aand Test dataset
train = pd.read_csv("exercise_02_train.csv")
test = pd.read_csv("exercise_02_test.csv")
 

# Plot the distrinution of target variable
train["y"].astype("int").plot.hist()
# This is unbalanced dataset because target variable(1) is on minority side

# Combine the train and Test dataset for analysis and cleaning purpose
comb = pd.concat([train,test], axis = 0, ignore_index=True)
comb.shape
comb.dtypes.value_counts()


############################################## Examine and Imputing the missing value ##########################################

# Create the function to find the mising value
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
print("Remaining missing values after imputing: ", sum(comb.isnull().sum()))

# Remove the Remaining missing values in combined dataset
comb = comb.dropna()
print("Remaining missing values after imputing: ", sum(comb.isnull().sum()))



############################################## Treatment on categorical features ############################################

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


# 3] convert categorical variables to numeric form using Label encoder and one hot encoded  
comb.dtypes.value_counts()
from sklearn.preprocessing import LabelEncoder

# label encoder
le = LabelEncoder()
count=0
for i in comb:
    if comb[i].dtypes=="object":  
        if len(list(comb[i].unique())) > 0:
            comb[i] = le.fit(comb[i]).transform(comb[i])
            count += 1
            
print('No. of columns were encoded: ', count)            

# One Hot Encoded
comb = pd.get_dummies(comb)
print("Training shape :", comb.shape)



"""# Align the train and test dataset 
train_labels = train["y"]

train, test = train.align(test, join = "inner", axis = 1)
train["y"] = train_labels
train.shape, test.shape"""

######################################### Treatment on continous variable ##################################################
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


#################3############################# Feature importance ##########################################################3
# Retrieve important features and sort them accordingly 
from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(n_estimators = 100, random_state= 0)
RF.fit(train.drop(columns="y"), train["y"])
feature_importance = RF.feature_importances_
feature_importance = pd.DataFrame({'feature': list(train.drop(columns="y").columns), 'importance': feature_importance}).sort_values('importance', ascending = False)
feature_importance.head(30)

# Remove the features having less importance from traiin and test
drop_zero = list(feature_importance[feature_importance.importance <=0.01].feature)
train = train.drop(columns = drop_zero)
test = test.drop(columns = drop_zero)
train.shape, test.shape


################################################ Random Forest Model #################################################

# Split the train dataset 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train.iloc[:,:-1], train.iloc[:,-1], train_size= 0.7, stratify = train.y, random_state = 0)

# Fit and Tunning the Random Forest model

from sklearn.model_selection import  RandomizedSearchCV
parameters = {'n_estimators': [100,200], 
               'max_features' : ['auto', 'sqrt', 'log2', None],
               'min_samples_split' : [2, 5, 10],
               'min_samples_leaf' : [1,2,4],
               'bootstrap' : [True,False],
               'criterion' : ["gini", "entropy"]
               }

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(random_state = 0)

Random_searchCV = RandomizedSearchCV(estimator = classifier, 
                           param_distributions = parameters, 
                           n_iter = 100,
                           cv = 10, 
                           n_jobs = -1,
                           verbose=5)

clf = Random_searchCV.fit(X_train, y_train)
best_accuracy = clf.best_score_
best_parameters = clf.best_params_
pred = clf.predict(X_test)
prob = clf.predict_proba(X_test)

# Analyze the performance of Random Forest model using confusion matrix and AUC
from sklearn.metrics import confusion_matrix, average_precision_score, auc, roc_curve
cm =confusion_matrix(y_test, pred)

average_precision = average_precision_score(y_test, prob[:,1])
print("Average precision Recall score: ",(average_precision))

fpr, tpr,threshold_ = roc_curve(y_test, prob[:,1])
roc_auc = auc(fpr, tpr)                
print("AUC score: ",(roc_auc))


# Prediction on test.csv
final_pred = clf.predict(test)

# save the result into results1.csv
test["y"] = final_pred
test.to_csv('results1.csv', index=False)

# Dump the model with python pickle
import pickle
results1 = "results1.pkl"
Random_forest_pkl = open(results1, 'wb')
pickle.dump(clf, Random_forest_pkl)
Random_forest_pkl.close()

# Load the model with python pickle
results1="results1.pkl"
Random_forest_pkl = open(results1, 'rb')
clf = pickle.load(Random_forest_pkl)
final_pred = clf.predict(test)
