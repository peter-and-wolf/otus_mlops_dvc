
from os import path
from typing import Tuple, Union

import pandas as pd
from numpy.typing import ArrayLike
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib


ORIGINAL_TRAIN_PATH = 'data/train.csv'
PROCESSED_TRAIN_PATH = 'features/train_preprocessed.pkl'
ORIGINAL_TEST_PATH = 'data/test.csv'
PROCESSED_TEST_PATH = 'features/test_preprocessed.pkl'
MODEL_PATH = 'models/classifier.pkl'


ModelType = Union[ 
  LogisticRegression, 
  DecisionTreeClassifier
]


def load_data(path: str, dropna: bool = False) -> Tuple[pd.DataFrame, pd.Series]:
  df = pd.read_csv(path, na_values='?')

  if dropna:
    df = df.dropna()
    
  X = df.drop(['id', 'is_rich'], axis=1)
  y = df['is_rich']
    
  return X, y


def preprocess(X: pd.DataFrame, tag: str):
  print(f'preprocerssing original {tag} data...\n')

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


def get_data(orig_path: str, proc_path: str, tag: str):
  if path.exists(proc_path):
    print(f'loading preprocessed {tag} data...\n')
    X, y = joblib.load(proc_path)
  else:
    X, y = load_data(orig_path)
    X = preprocess(X, tag)
    joblib.dump((X, y), proc_path)
  
  return X, y
 

def validate_model(
    model: ModelType, 
    X: ArrayLike, 
    y_true: ArrayLike, 
    tag: str
    ):
  
  y_pred = model.predict(X)

  accuracy = accuracy_score(y_pred, y_true)
  precision = precision_score(y_pred, y_true)
  recall = recall_score(y_pred, y_true)
  f_score=f1_score(y_pred, y_true)
  print(f'{tag} accuracy: {accuracy}')
  print(f'{tag} precision: {precision}')
  print(f'{tag} recall: {recall}')
  print(f'{tag} f1_score: {f_score}')
  print('\n')


def train_model(X, y) -> ModelType:
  X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
  
  model = LogisticRegression(max_iter=1_000)
  model.fit(X_train, y_train)
  validate_model(model, X_val, y_val, 'train')
  
  return model

def run():
  X, y = get_data(ORIGINAL_TRAIN_PATH, PROCESSED_TRAIN_PATH, 'train')
  
  model = train_model(X, y)
  joblib.dump(model, MODEL_PATH)

  X_test, y_test = get_data(ORIGINAL_TEST_PATH, PROCESSED_TEST_PATH, 'test')
  validate_model(model, X_test, y_test, 'test') # type: ignore


if __name__ == '__main__':
    run()


