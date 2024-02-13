
from os import path
from typing import Tuple

import pandas as pd
from numpy.typing import ArrayLike
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.base import ClassifierMixin
import joblib


ORIGINAL_DATA_PATH = 'data/train.csv'
PREPROCESSED_DATA_PATH = 'data/preprocessed.pkl'
MODEL_PATH = 'weights/classifier.pkl'


def load_data(path: str, dropna: bool = False) -> Tuple[pd.DataFrame, pd.Series]:
  df_train = pd.read_csv(path, na_values='?')

  if dropna:
    df_train = df_train.dropna()
    
  X = df_train.drop(['id', 'is_rich'], axis=1)
  y = df_train['is_rich']
    
  return X, y


def preprocess(X: pd.DataFrame):
  print('preprocerssing original data...')

  categorical = X.select_dtypes(include=['object', 'bool']).columns
  numerical = X.select_dtypes(include=['int64', 'float64']).columns
  
  cat_proc = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoding', OrdinalEncoder()),
    #('encoding', OneHotEncoder(handle_unknown='ignore')),
  ])
    
  transformer = ColumnTransformer([ 
    ('categorical', cat_proc, categorical),
    ('scaling', MinMaxScaler(), numerical),
    #('scaling', StandardScaler(), numerical)
  ])

  return transformer.fit_transform(X)
 

def print_metrics(y1: ArrayLike, y2: ArrayLike):
  accuracy = accuracy_score(y1, y2)
  precision = precision_score(y1, y2)
  recall = recall_score(y1, y2)
  f_score=f1_score(y1, y2)
  print(f'accuracy: {accuracy}')
  print(f'precision: {precision}')
  print(f'recall: {recall}')
  print(f'f1_score: {f_score}')


def train_model(X, y) -> ClassifierMixin:
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  
  model = LogisticRegression(max_iter=1_000)
  model.fit(X_train, y_train)

  y_pred = model.predict(X_test)
  print_metrics(y_test, y_pred)

  return model

def run():
  if path.exists(PREPROCESSED_DATA_PATH):
    print('loading preprocessed data...')
    X, y = joblib.load(PREPROCESSED_DATA_PATH)
  else:
    X, y = load_data(ORIGINAL_DATA_PATH)
    X = preprocess(X)
    joblib.dump((X, y), PREPROCESSED_DATA_PATH)
  
  model = train_model(X, y)
  joblib.dump(model, MODEL_PATH)


if __name__ == '__main__':
    run()


