
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, Normalizer, LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
from hyperopt import hp, fmin, Trials, tpe


# reading dataset
def read_csv(path):
    if path is not None:
        data = pd.read_csv(path)
        print(data.head())
    else:
        print('None dataset')


data = pd.read_csv("Safra_2018-2019.csv")


# class names
class_names = ['saudável', 'pesticidas', 'outros']


# drop cols
def drop_cols(*cols):
    if cols:
        data.drop(*cols, axis=1, inplace=True)
        print('Cols droped: {}'.format(*cols))
    else:
        pass


# cols to drop
cols_to_drop = ['Unnamed: 0', 'Identificador_Agricultor']
drop_cols(cols_to_drop)


# features and target
X = data.drop('dano_na_plantacao', axis=1)
y = data['dano_na_plantacao']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42)


mdl = XGBClassifier(random_state=42)
mdl.fit(X_train, y_train)
y_pred = mdl.predict(X_test)


# feature scaling
def feature_modeling(data, num_cols, scaler=StandardScaler()):

    if num_cols:
        for col in num_cols:
            data[col] = scaler.fit_transform(data[[col]])
    else:
        pass


cols = ['Estimativa_de_Insetos',
        'Tipo_de_Cultivo',
        'Tipo_de_Solo',
        'Categoria_Pesticida',
        'Doses_Semana',
        'Semanas_Utilizando',
        'Semanas_Sem_Uso',
        'Temporada']

feature_modeling(data, num_cols=cols, scaler=StandardScaler())


# Tuning hyperparameters

def objetive(params):
    return -accuracy_score(y_test, y_pred)


spaces_xgboost = {'n_estimators': hp.randint('n_estimators', 1000),
                  'learning_rate': hp.loguniform('learning_rate', 0.0001, 0.1),
                  'max_depth': hp.randint('max_depth', 25),
                  'min_child_weight': hp.uniform('min_child_weight', 0, 20),
                  'reg_lambda': hp.uniform('reg_lambda', 0.001, 3),
                  'reg_alpha': hp.uniform('reg_alpha', 0.001, 3),
                  'gamma': hp.uniform('gamma', 0, 10),
                  'max_delta_step': hp.uniform('max_delta_step', 0, 10),
                  'max_leaves': hp.randint('max_leaves', 30),
                  'colsample_bytree': hp.uniform('colsample_bytree', 0.1, 1.0)
                  }


trials = Trials()
tuning = fmin(objetive, spaces_xgboost, algo=tpe.suggest,
              max_evals=50, trials=trials)


# XGBoost after Tuning
def model_tuning(**params):

    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return print(classification_report(y_test, y_pred, target_names=class_names))


# XGBoost with best parameters
clf = model_tuning(**tuning, objective='multi:softmax', random_state=42)


# salving model
file_name = 'xgb_clf.pkl'
pickle.dump(clf, open(file_name, "wb"))
