import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
import joblib
import os
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
import tensorflow as tf
from skopt import BayesSearchCV


def save_model(model, model_name='new_model', directory='model'):
    """
    Saves the trained model to the specified directory.
    :param model: The model object to be saved
    :param model_name: name of the model file to save (without suffix)
    :param directory: The name of the directory where the model will be saved.
    """
    # Ensure that the model save directory exists, and create it if it doesn't
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
            print(f"Directory '{directory}' was created.")
        except Exception as e:
            print(f"Failed to create directory '{directory}'. Error: {e}")
            return

    # Define the full path where the model is saved
    model_path = os.path.join(directory, f"{model_name}.joblib")

    # Trying to save the model
    try:
        joblib.dump(model, model_path)
        print(f"Model saved successfully at {model_path}")
    except Exception as e:
        print(f"Failed to save the model. Error: {e}")






# Load data
train_data = pd.read_json('dataset/train_c.json')
val_data = pd.read_json('dataset/validation_c.json')
test_data = pd.read_json('dataset/test_c.json')

# Data cleansing, e.g. handling of missing values
imputer = SimpleImputer(strategy='constant', fill_value=0)
train_data['time_series_similarity'] = imputer.fit_transform(train_data[['time_series_similarity']])
val_data['time_series_similarity'] = imputer.transform(val_data[['time_series_similarity']])
test_data['time_series_similarity'] = imputer.transform(test_data[['time_series_similarity']])

# Defining features and labels
X_train = train_data[['sbert_similarity', 'sentiment_match', 'time_series_similarity']]
y_train = train_data['ood']
X_val = val_data[['sbert_similarity', 'sentiment_match', 'time_series_similarity']]
y_val = val_data['ood']
X_test = test_data[['sbert_similarity', 'sentiment_match', 'time_series_similarity']]
y_test = test_data['ood']


# Suppose X_train, y_train are your training data, oversampled.
smote = SMOTE(random_state=20)
X_res, y_res = smote.fit_resample(X_train, y_train)

# # Training logistic regression models
# model = LogisticRegression(
#     solver='lbfgs',
#     max_iter=50,  # 迭代次数
#     C=0.5,        # 正则化强度
#     penalty='l2',  # 使用L2正则化
#     tol=1e-4      # 设置更严格的停止条件
# )
#
#
# model.fit(X_res, y_res)


# Define the model and also set the category weights
rf = RandomForestClassifier(random_state=42, class_weight={0: 1, 1: 1.5})

# Setting the parameter space for Bayesian optimised search
param_space = {
    'max_depth': (3, 30),  # Can be an integer or a (None, integer) tuple representing the maximum depth of the tree
    'n_estimators': (10, 1000),  # Number of trees
    'min_samples_split': (2, 100),  # Minimum number of samples required for node segmentation
    'min_samples_leaf': (1, 50),  # Minimum number of samples required for leaf nodes
    'max_features': ['sqrt', 'log2', None]  # Number of features to consider when finding the optimal segmentation
}

# Instantiating Bayesian Search Objects
opt = BayesSearchCV(rf, param_space, n_iter=32, scoring="accuracy", cv=3, verbose=3, random_state=42, n_jobs=-1)

# conduct a search
opt.fit(X_res, y_res)

# View the best combination of parameters
print("Best parameters found: ", opt.best_params_)


# Training Models
opt.fit(X_res, y_res)

# Optimal parameters and models
print("Best parameters:", opt.best_params_)
best_model = opt.best_estimator_

# Prediction using optimal models
y_pred = best_model.predict(X_test)

# assessment model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


# verification model
y_pred_val = best_model.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, y_pred_val))
print("Validation Report:", classification_report(y_val, y_pred_val))

# test model
y_pred_test = best_model.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred_test))
print("Test Report:", classification_report(y_test, y_pred_test))


# Creating a model save path
model_directory = 'model'
if not os.path.exists(model_directory):
    os.makedirs(model_directory)
model_path = os.path.join(model_directory, f'model.joblib')
# Save the model
save_model(best_model)



