FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py analysis.py index.html ./
ENV HOST=0.0.0.0
EXPOSE 8501
CMD ["python", "app.py"]
