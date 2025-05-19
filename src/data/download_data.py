import os
import sqlite3
import pandas as pd
import zipfile
import json

def check_kaggle_auth():
    """ Tra xem người dùng đã cấu hình thông tin xác thực Kaggle hay chưa."""
    
    kaggle_dir = os.path.expanduser('~/.kaggle')
    kaggle_file = os.path.join(kaggle_dir, 'kaggle.json')
    
    if not os.path.exists(kaggle_file):
        print("Kaggle API credentials not found!")
        print("Please create an API token from kaggle.com and place it at:")
        print(f"  {kaggle_file}")
        return False
    
    # Verify the file contains valid JSON with expected keys
    try:
        with open(kaggle_file, 'r') as f:
            creds = json.load(f)
        if 'username' in creds and 'key' in creds:
            return True
        else:
            print("Kaggle credentials file exists but appears to be invalid.")
            return False
    except Exception as e:
        print(f"Error reading Kaggle credentials: {e}")
        return False

def download_data():
    """ Tải dữ liệu từ Kaggle về thư mục data """
    
    if not os.path.exists("data/database.sqlite"):
        # Check Kaggle authentication first
        if not check_kaggle_auth():
            raise Exception("Kaggle authentication not configured correctly. Cannot download data.")
        
        # Make sure the data directory exists
        os.makedirs("data", exist_ok=True)
        
        # Download the dataset using kaggle command
        print("Downloading dataset from Kaggle...")
        os.system("kaggle datasets download -d hugomathien/soccer -p data/")
        
        # Extract using Python's zipfile module instead of unzip command
        print("Extracting zip file...")
        with zipfile.ZipFile("data/soccer.zip", 'r') as zip_ref:
            zip_ref.extractall("data/")
        
        # Remove the zip file after extraction
        print("Removing zip file...")
        os.remove("data/soccer.zip")
        print("Download complete!")
    return

def extract_sqlite_to_csv(sqlite_path, output_dir):
    """ Trích xuất tất cả các bảng từ file SQLite và lưu thành các file CSV. """
    
    if not os.path.exists(sqlite_path):
        raise FileNotFoundError(f"Không tìm thấy file SQLite tại {sqlite_path}")

    os.makedirs(output_dir, exist_ok=True)
    
    conn = sqlite3.connect(sqlite_path)
    cursor = conn.cursor()

    # Lấy danh sách tên bảng trong database
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    exclude_tables = {"sqlite_sequence"}
    tables = [t[0] for t in cursor.fetchall() if t[0] not in exclude_tables]

    for table_name in tables:
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        csv_path = os.path.join(output_dir, f"{table_name}.csv")
        df.to_csv(csv_path, index=False)
        print(f"[OK] Đã lưu bảng `{table_name}` vào `{csv_path}`")

    conn.close()
    print("[DONE] Tất cả bảng đã được trích xuất thành CSV.")

if __name__ == "__main__":
    # Kiểm tra xem Kaggle API đã được cấu hình chưa
    check_kaggle_auth()
    # Tải dữ liệu từ Kaggle
    download_data()
    # Chuyển đổi dữ liệu từ định dạng SQLite sang CSV
    extract_sqlite_to_csv("data/database.sqlite", "data/raw")