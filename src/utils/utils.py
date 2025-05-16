import numpy as np
import pandas as pd
import os
from pathlib import Path

def seed_everything(seed=42):
    import random, os
    random.seed(seed)
    np.random.seed(seed)
    pass

def check_missing_values(df, df_name):
    '''Kiểm tra và in ra các cột có giá trị thiếu'''
    
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        print(f"Giá trị thiếu trong {df_name}: \n{missing}\n")
    else:
        print(f"Không có giá trị thiếu trong {df_name}.\n")

def inspect_data(df: pd.DataFrame, name: str = "DataFrame"):
    '''Kiểm tra thông tin và hiển thị vài hàng của DataFrame'''
    
    print(f"\n>>> Kiểm tra {name}:")
    print("Kích thước:", df.shape)
    print("Kiểu dữ liệu:\n", df.dtypes)
    print("3 hàng đầu:\n", df.head(3), "\n")

def save_split_data(train, test, train_file='data/train.csv', test_file='data/test.csv'):
    '''Lưu dữ liệu sau khi split thành file CSV.'''
    
    train.to_csv(train_file, index=False)
    test.to_csv(test_file, index=False)
    print(f"Đã lưu dữ liệu: {train_file}, {test_file}")
    
def save_intermediate_data(data, file_name, processed_dir='data/processed'):
    '''Lưu dữ liệu trung gian ở mỗi bước xử lý'''
    
    os.makedirs(processed_dir, exist_ok=True)

    # Nếu data là một DataFrame duy nhất
    if isinstance(data, pd.DataFrame):
        file_path = Path(processed_dir) / file_name
        data.to_csv(file_path, index=False)
        print(f"Đã lưu dữ liệu trung gian: {file_path}")

    # Nếu data là tuple hoặc list (như train, test)
    elif isinstance(data, (tuple, list)):
        if len(data) != len(file_name):
            raise ValueError("Số lượng file_name phải bằng số lượng DataFrame trong data.")
        for df, fname in zip(data, file_name):
            file_path = Path(processed_dir) / fname
            df.to_csv(file_path, index=False)
            print(f"Đã lưu dữ liệu trung gian: {file_path}")

    else:
        raise ValueError("Dữ liệu trung gian phải là pandas DataFrame hoặc tuple/list của DataFrame.")