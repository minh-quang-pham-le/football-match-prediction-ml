import numpy as np

def seed_everything(seed=42):
    import random, os
    random.seed(seed)
    np.random.seed(seed)
    pass