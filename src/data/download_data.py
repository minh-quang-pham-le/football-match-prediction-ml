import os
import sqlite3
import pandas as pd
from typing import List

def download_data():
    """ Tải dữ liệu từ Kaggle về thư mục data """
    
    if not os.path.exists("data/database.sqlite"):
        os.system("kaggle datasets download -d hugomathien/soccer -p data/")
        os.system("unzip data/soccer.zip -d data/")
        os.remove("data/soccer.zip")

def extract_sqlite_to_csv(sqlite_path="data/database.sqlite", output_dir="data/raw"):
    """ Trích xuất tất cả các bảng từ file SQLite và lưu thành các file CSV. """
    
    if not os.path.exists(sqlite_path):
        raise FileNotFoundError(f"Không tìm thấy file SQLite tại {sqlite_path}")

    os.makedirs(output_dir, exist_ok=True)
    
    conn = sqlite3.connect(sqlite_path)
    cursor = conn.cursor()

    # Lấy danh sách tên bảng trong database
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    for table_name in tables:
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        csv_path = os.path.join(output_dir, f"{table_name}.csv")
        df.to_csv(csv_path, index=False)
        print(f"[OK] Đã lưu bảng `{table_name}` vào `{csv_path}`")

    conn.close()
    print("[DONE] Tất cả bảng đã được trích xuất thành CSV.")