# Image nhẹ, ổn định
FROM python:3.10-slim

# Không tạo file .pyc và hiển thị log ngay
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Cài pip packages
COPY docker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Đặt thư mục làm việc
WORKDIR /app
COPY . /app

# Tạo user không phải root (an toàn hơn)
RUN adduser --disabled-password --gecos "" appuser && chown -R appuser /app
USER appuser

# Mở port cho Gradio (nếu bạn chạy Gradio)
EXPOSE 7860

# Điểm khởi chạy: bạn có thể đổi thành run_pipeline.py nếu muốn
CMD ["python", "demo/app.py"]