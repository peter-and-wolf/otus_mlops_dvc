import time
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression # type: ignore
from sklearn.multiclass import OneVsRestClassifier  # type: ignore
from sklearn.metrics import accuracy_score # type: ignore

import config as cfg


def scale(train_in_path: str, 
          test_in_path: str,
          train_out_path: str,
          test_out_path: str) -> None:

  t1 = time.time()
  print('\nscaling...')
  
  train_df = pd.read_csv(train_in_path, header=None, dtype=float)
  test_df = pd.read_csv(test_in_path, header=None, dtype=float)


  train_mean = train_df.values[:, 1:].mean()
  train_std = train_df.values[:, 1:].std()

  print(f'train_mean={train_mean}, train_std={train_std}')

  train_df.values[:, 1:] -= train_mean
  train_df.values[:, 1:] /= train_std
  test_df.values[:, 1:] -= train_mean
  test_df.values[:, 1:] /= train_std
  
  np.save(train_out_path, train_df)
  np.save(test_out_path, test_df)

  print(f'done for {time.time() - t1:.2f}s')


def train(train_path: str, model_path) -> None:

  t1 = time.time()
  print('\ntraining...')

  np.random.seed(cfg.RANDOM_SEED)

  train_data = np.load(train_path)

  model = OneVsRestClassifier(LogisticRegression(solver='newton-cg'), n_jobs=6)
  model.fit(train_data[:, 1:], train_data[:, 0])

  with open(model_path, 'wb') as f:
    pickle.dump(model, f)
    print('model has been saved')

  print(f'done for {time.time() - t1:.2f}s')


def eval(test_path: str, model_path: str) -> None:

  print('\nevaluation...')

  test_data = np.load(test_path)

  with open(model_path, 'rb') as f:
    model = pickle.load(f)
    print('model has been loaded')

  acc = accuracy_score(test_data[:, 0], model.predict(test_data[:, 1:]))

  print(f'accuracy on test: {acc}')


if __name__ == '__main__':
  scale(
    train_in_path=cfg.TRAIN_DATA_CSV_PATH,
    test_in_path=cfg.TEST_DATA_CSV_PATH,
    train_out_path=cfg.TRAIN_DATA_NPY_PATH,
    test_out_path=cfg.TEST_DATA_NPY_PATH
  )
  train(
    train_path=cfg.TRAIN_DATA_NPY_PATH,
    model_path=cfg.MODEL_PKL_PATH
  )
  eval(
    test_path=cfg.TEST_DATA_NPY_PATH,
    model_path=cfg.MODEL_PKL_PATH
  )



