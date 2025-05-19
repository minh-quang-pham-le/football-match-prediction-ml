import numpy as np
import pandas as pd
import os
from pathlib import Path

def seed_everything(seed=42):
    import random, os
    random.seed(seed)
    np.random.seed(seed)
    pass