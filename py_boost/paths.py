import os
from pathlib import Path

PROJECT_PATH = str(Path(__file__).parent.parent)
EXPERIMENTS_PATH = os.path.join(PROJECT_PATH, 'experiments')
EXPERIMENTS_DATA_PATH = os.path.join(EXPERIMENTS_PATH, 'data')

if __name__ == '__main__':
    print(os.path.join(EXPERIMENTS_DATA_PATH, 'mediamill', 'mediamill.arff'))
    print(EXPERIMENTS_DATA_PATH + f'/age_pred/fold_{3}/emb_test.csv')
