# Digital Twin thang máy – chạy ở mọi nơi có Docker (Render, Hugging Face Spaces, máy chủ trường…)
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PORT=7860
EXPOSE 7860
# 1 tiến trình (phiên trực tiếp nằm trong bộ nhớ) + nhiều luồng
CMD gunicorn -w 1 --threads 8 --timeout 180 -b 0.0.0.0:${PORT} web_app:app
