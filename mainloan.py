# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
#%matplotlib inline
import seaborn as sns
sns.set_style('dark')
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import warnings
from sklearn.exceptions import ConvergenceWarning

# Suppress all warnings
warnings.filterwarnings("ignore")

# Suppress convergence warnings specific to Logistic Regression
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# reading datasets
df_train = pd.read_csv('loan-train.csv')
df_test = pd.read_csv('loan-test.csv')

#prnt shape od datasets
print('Train dataset shape:', df_train.shape)
print('-------------------------------------------------------------------------------------------')
print('Test dataset shape:', df_test.shape)

df_train.head()
df_train.describe()

# plotting crosstab
pd.crosstab(df_train['Credit_History'], df_train['Loan_Status'], margins =True)

# checking for outliers
plt.figure(figsize=(12, 8))
df_train.boxplot()

# plotting boxplot for training data
plt.title('Boxplot of training dateset')


plt.figure(figsize=(12, 8))
df_test.boxplot()

# plotting boxplot for test data
plt.title('Boxplot of test dateset')


# Show the plot
plt.show()

plt.figure(figsize=(7, 5))
df_train['ApplicantIncome'].hist(bins=20)
plt.xlabel('Applicant Income')
plt.ylabel('Frequency')
plt.title('Histogram of Applicant Income')

plt.figure(figsize=(7, 5))
df_train['CoapplicantIncome'].hist(bins=20)
plt.xlabel('Coapplicant Income')
plt.ylabel('Frequency')
plt.title('Histogram of Coapplicant Income')

plt.figure(figsize=(7, 5))
df_train['LoanAmount'].hist(bins=20)
plt.xlabel('Loan Amount')
plt.ylabel('Frequency')
plt.title('Histogram of Loan Amount')

plt.figure(figsize=(7, 5))
df_train['Loan_Amount_Term'].hist(bins=20)
plt.xlabel('Loan_Amount_Term')
plt.ylabel('Frequency')
plt.title('Histogram of Loan_Amount_Term')

plt.figure(figsize=(7, 5))
df_train['Credit_History'].hist(bins=20)
plt.xlabel('Credit_History')
plt.ylabel('Frequency')
plt.title('Histogram of Credit_History')

plt.show()

plt.figure(figsize=(7, 5))
df_train.boxplot(column='ApplicantIncome', by='Education')
plt.show()

plt.figure(figsize=(7, 5))
df_train['ApplicantIncome_log'].hist(bins=20) 


plt.figure(figsize=(7, 5))
df_train['LoanAmount_log'].hist(bins=20) 

plt.figure(figsize=(7, 5))
df_train['Loan_Amount_Term-log'].hist(bins=20) 

df_train.isnull().sum()

df_train['Gender'].fillna(df_train['Gender'].mode()[0], inplace=True) # Mode
df_test['Gender'].fillna(df_test['Gender'].mode()[0], inplace=True) # Mode

df_train['Married'].fillna(df_train['Married'].mode()[0], inplace=True) # Mode
df_test['Married'].fillna(df_test['Married'].mode()[0], inplace=True) # Mode

df_train['Dependents'].fillna(df_train['Dependents'].mode()[0], inplace=True) # Mode
df_test['Dependents'].fillna(df_test['Dependents'].mode()[0], inplace=True) # Mode

df_train['Self_Employed'].fillna(df_train['Self_Employed'].mode()[0], inplace=True) # Mode
df_test['Self_Employed'].fillna(df_test['Self_Employed'].mode()[0], inplace=True) # Mode

df_train['LoanAmount'].fillna(df_train['LoanAmount'].mean(), inplace=True) # Mean
df_test['LoanAmount'].fillna(df_test['LoanAmount'].mean(), inplace=True) # Mean

df_train['Loan_Amount_Term'].fillna(df_train['Loan_Amount_Term'].mode()[0], inplace=True) # Mode
df_test['Loan_Amount_Term'].fillna(df_test['Loan_Amount_Term'].mode()[0], inplace=True) # Mode

df_train['Credit_History'].fillna(df_train['Credit_History'].mode()[0], inplace=True) # Mode
df_test['Credit_History'].fillna(df_test['Credit_History'].mode()[0], inplace=True) # Mode

df_train['LoanAmount_log'].fillna(df_train['LoanAmount_log'].mean(), inplace=True) # mean


df_train['Loan_Amount_Term-log'].fillna(df_train['Loan_Amount_Term-log'].mode()[0], inplace=True) # mode

df_train['TotalIncome'] = df_train['ApplicantIncome'] + df_train['CoapplicantIncome']
df_train['TotalIncome_log'] = np.log(df_train['TotalIncome'])

df_train['TotalIncome_log'].hist(bins=20)

df_train.Loan_Status = df_train.Loan_Status.replace({"Y": 1, "N" : 0})


df_train.Gender = df_train.Gender.replace({"Male": 1, "Female" : 0})
df_test.Gender = df_test.Gender.replace({"Male": 1, "Female" : 0})

df_train.Married = df_train.Married.replace({"Yes": 1, "No" : 0})
df_test.Married = df_test.Married.replace({"Yes": 1, "No" : 0})

df_train.Self_Employed = df_train.Self_Employed.replace({"Yes": 1, "No" : 0})
df_test.Self_Employed = df_test.Self_Employed.replace({"Yes": 1, "No" : 0})

from sklearn.preprocessing import LabelEncoder
feature_col = ['Property_Area','Education', 'Dependents']
le = LabelEncoder()
for col in feature_col:
    df_train[col] = le.fit_transform(df_train[col])

df_train.plot(figsize=(18, 8))

plt.show()

plt.figure(figsize=(18, 6))
plt.title("Relation Between Applicant Income vs Loan Amount ")

sns.scatterplot(data=df_train, x='ApplicantIncome', y='LoanAmount', hue='Loan_Status', palette='Set1')
plt.xlabel("Applicant Income")
plt.ylabel("Loan Amount")
plt.grid()
plt.show()

plt.figure(figsize=(12, 6))
plt.title("Loan Application Amount ")

sns.lineplot(data=df_train, x='Loan_Status', y='LoanAmount')
plt.xlabel("Loan Status")
plt.ylabel("Loan Amount")
plt.grid()
plt.show()

plt.figure(figsize=(12,8))
sns.heatmap(df_train.corr(), cmap='coolwarm', annot=True, fmt='.1f', linewidths=.1)
plt.show()


features = ['Married', 'Dependents', 'Education', 'Self_Employed', 'Credit_History', 'Property_Area', 'ApplicantIncome', 'LoanAmount', 'Loan_Amount_Term','ApplicantIncome_log', 'LoanAmount_log', 'Loan_Amount_Term-log']
X = df_train[features]
target = ['Loan_Status']
y = df_train[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# instanciating logistic model
logistic_model = LogisticRegression()

# fitting training data
logistic_model.fit(X_train, y_train)

#prediction 
pred1 = logistic_model.predict(X_test)

# getting accuracy
test_accuracy = accuracy_score(y_test, pred1)

# printing accuracy score
print('logistic regression accuracy:', test_accuracy)

# Define the parameter grid to search over
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Regularization parameter
    'penalty': ['l1', 'l2'],  # Regularization penalty
    'solver': ['liblinear'],  # Optimization algorithm
    'max_iter': [100, 200, 300]  # Maximum number of iterations
}



# Instantiate GridSearchCV
grid_search = GridSearchCV(logistic_model, param_grid, cv=5, scoring='accuracy')

# Fit GridSearchCV to the training data
grid_search.fit(X_train, y_train)

# Get the best parameters and the best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_


# Get the best model
best_model = grid_search.best_estimator_

# Evaluate the best model on the testing set
pred_tuned1 = best_model.predict(X_test)
test_accuracy_tuned = accuracy_score(y_test, pred_tuned1)
print('accuracy score for tuned logistic regression:', test_accuracy_tuned)

# Instantiate DecisionTreeClassifier
decision_tree = DecisionTreeClassifier()

# Fit the model to the training data
decision_tree.fit(X_train, y_train)

# Make predictions on the training and testing sets
pred2 = decision_tree.predict(X_test)

# getting accuracy
test_accuracy_tree = accuracy_score(y_test, pred2)

print('Decision tree accuracy:', test_accuracy_tree)

# Define the parameter grid to search over
param_grid_tree = {
    'criterion': ['gini', 'entropy'],  # Split criterion
    'max_depth': [None, 10, 20, 30, 40, 50],  # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4],  # Minimum number of samples required to be at a leaf node
    'max_features': ['auto', 'sqrt', 'log2']  # Number of features to consider when looking for the best split
}



# Instantiate GridSearchCV
grid_search_tree = GridSearchCV(decision_tree, param_grid_tree, cv=5, scoring='accuracy')

# Fit GridSearchCV to the training data
grid_search_tree.fit(X_train, y_train)

# Get the best parameters and the best score
best_params_tree = grid_search.best_params_
best_score_tree = grid_search.best_score_


# Get the best model
best_decision_tree = grid_search.best_estimator_

# getting tree prediction
pred_tuned2 = best_decision_tree.predict(X_test)

# Evaluate the best model on the testing set
test_accuracy_tree = accuracy_score(y_test, pred_tuned2)
print("Testing Accuracy od tuned decision tree model:", test_accuracy_tree)

# instantiating gaussiannb model
gnb = GaussianNB()

# Fit the model to the training data
gnb.fit(X_train, y_train)

# Make predictions on the testing data
pred3 = gnb.predict(X_test)

# accuracy score of guassian nb
test_accuracy_guassian = accuracy_score(y_test, pred3)
print("Testing Accuracy of guassiannb model:", test_accuracy_guassian)

# Instantiate XGBClassifier
xgb_classifier = XGBClassifier()

# Fit the model to the training data
xgb_classifier.fit(X_train, y_train)

# Make predictions on the testing data
pred4 = xgb_classifier.predict(X_test)

# Calculate accuracy
test_accuracy_xgboost = accuracy_score(y_test, pred4)
print("Accuracy score for xgboost model:", test_accuracy_xgboost)

# Define the parameter grid to search over
param_grid = {
    'n_estimators': [100, 200, 300],  # Number of boosting rounds
    'learning_rate': [0.01, 0.1, 0.2],  # Step size shrinkage used in update to prevent overfitting
    'max_depth': [3, 5, 7],  # Maximum depth of a tree
    'subsample': [0.5, 0.7, 1.0],  # Subsample ratio of the training instances
    'colsample_bytree': [0.5, 0.7, 1.0],  # Subsample ratio of columns when constructing each tree
}



# Instantiate GridSearchCV
grid_search_xgboost = GridSearchCV(xgb_classifier, param_grid, cv=5, scoring='accuracy')

# Fit GridSearchCV to the training data
grid_search_xgboost.fit(X_train, y_train)

# Get the best parameters and the best score
best_params_xg = grid_search_xgboost.best_params_
best_score_xg = grid_search_xgboost.best_score_

# Get the best model
best_xgb_classifier = grid_search_xgboost.best_estimator_

# Evaluate the best model on the testing set
pred_tuned4 = best_xgb_classifier.predict(X_test)

test_accuracy_xgboost_tuned = accuracy_score(y_test, pred_tuned4)
print("Accuracy score for tuned xgboost model:", test_accuracy_xgboost_tuned)

# Instantiate RandomForestClassifier
random_forest = RandomForestClassifier()

# Fit the model to the training data
random_forest.fit(X_train, y_train)

# Make predictions on the testing data
pred5 = random_forest.predict(X_test)

# accurscy score
test_accuracy_forest = accuracy_score(y_test, pred5)
print('Accuracy score for random forest model:', test_accuracy_forest)

# Define the parameter grid to search over
param_grid = {
    'n_estimators': [100, 200, 300],  # Number of trees in the forest
    'max_depth': [None, 10, 20, 30],  # Maximum depth of the trees
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4],  # Minimum number of samples required to be at a leaf node
    'max_features': ['auto', 'sqrt', 'log2']  # Number of features to consider when looking for the best split
}


# Instantiate GridSearchCV
grid_search_forest = GridSearchCV(random_forest, param_grid, cv=5, scoring='accuracy')

# Fit GridSearchCV to the training data
grid_search_forest.fit(X_train, y_train)

# Get the best parameters and the best score
best_params_forest = grid_search_forest.best_params_
best_score_forest = grid_search_forest.best_score_



# Get the best model
best_random_forest = grid_search_forest.best_estimator_

#predict with best nmodel
pred_tuned5 = best_random_forest.predict(X_test)
# Evaluate the best model on the testing set
test_accuracy_forest_tuned = accuracy_score(y_test, pred_tuned5)
print("Accuracy score for tuned random forest:", test_accuracy_forest_tuned)

accuracy_scores =[test_accuracy, test_accuracy_tuned, test_accuracy_tree, test_accuracy_tree, test_accuracy_guassian, test_accuracy_xgboost, test_accuracy_xgboost_tuned, test_accuracy_forest, test_accuracy_forest_tuned]
models = ['LogisticRegression', 'Gridsearch tuned LogisticRegression', 'DecisionTreeClassifier', 'Gridsearch tuned DecisionTreeClassifier', 'GaussianNB', 'XGBoost', 'Gridsearch tuned XGBoost', 'RandomForestClassifier', 'Gridsearch tuned RandomForestClassifier']

# Create a dataframe
df = pd.DataFrame({
    'Model': models,
    'Accuracy Score': accuracy_scores
})

# Display the dataframe
df.head(9)

df_test['ApplicantIncome_log'] = np.log(df_test['ApplicantIncome'])
df_test['LoanAmount_log'] = np.log(df_test['LoanAmount'])
df_test['Loan_Amount_Term-log'] = np.log(df_test['Loan_Amount_Term'])
df_test['LoanAmount_log'].fillna(df_test['LoanAmount_log'].mean(), inplace=True) # mean
df_test['Loan_Amount_Term-log'].fillna(df_test['Loan_Amount_Term-log'].mode()[0], inplace=True) # mode
df_test['TotalIncome'] = df_test['ApplicantIncome'] + df_test['CoapplicantIncome']
df_test['TotalIncome_log'] = np.log(df_test['TotalIncome'])

feature_col1 = ['Property_Area','Education', 'Dependents']
le = LabelEncoder()
for col in feature_col1:
    df_test[col] = le.fit_transform(df_test[col])

features = ['Married', 'Dependents', 'Education', 'Self_Employed', 'Credit_History', 'Property_Area', 'ApplicantIncome', 'LoanAmount', 'Loan_Amount_Term','ApplicantIncome_log', 'LoanAmount_log', 'Loan_Amount_Term-log']
x_test = df_train[features]

best_random_forest.predict(x_test)