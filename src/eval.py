import pickle
import json

import numpy as np
from sklearn.metrics import accuracy_score # type: ignore

import config as cfg


def eval(test_path: str, model_path: str, metrics_path) -> None:

  print('\nevaluation...')

  test_data = np.load(test_path)

  with open(model_path, 'rb') as f:
    model = pickle.load(f)
    print('model has been loaded')

  acc = accuracy_score(test_data[:, 0], model.predict(test_data[:, 1:]))

  print(f'accuracy on test: {acc}')

  with open(metrics_path, 'w') as f:
    json.dump({'accuracy': acc}, f)


if __name__ == '__main__':
  eval(
    test_path=cfg.TEST_DATA_NPY_PATH,
    model_path=cfg.MODEL_PKL_PATH,
    metrics_path=cfg.EVAL_METRICS_PATH
  )