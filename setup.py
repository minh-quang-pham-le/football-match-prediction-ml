import os
import subprocess
import sys
import platform

def install():
    print("Tạo môi trường ảo...")
    if not os.path.exists("venv"):
        subprocess.run([sys.executable, "-m", "venv", "venv"])

    # Xác định OS để biết cách activate
    system = platform.system()
    if system == "Windows":
        activate_cmd = ".\\venv\\Scripts\\activate &&"
    else:
        activate_cmd = "source ./venv/bin/activate &&"

    # Install requirements
    print("Cài đặt các thư viện...")
    command = f"{activate_cmd} pip install -r docker/requirements.txt"
    subprocess.call(command, shell=True)

    print("Môi trường đã sẵn sàng!")
    print("Giờ bạn có thể chạy: python run_pipeline.py")

if __name__ == "__main__":
    install()