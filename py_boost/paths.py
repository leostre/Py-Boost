import os
from pathlib import Path

PROJECT_PATH = str(Path(__file__).parent.parent)
EXPERIMENTS_PATH = os.path.join(PROJECT_PATH, 'experiments')
EXPERIMENTS_DATA_PATH = os.path.join(EXPERIMENTS_PATH, 'data')
