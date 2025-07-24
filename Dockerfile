# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Cài các thư viện cần thiết
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# copy code và model vào container
COPY . /app

# streamlit default port
EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
