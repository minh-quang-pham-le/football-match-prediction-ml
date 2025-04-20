import pandas as pd

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    # xử lý thiếu, mã hóa,...
    return df