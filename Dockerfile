# ---- 1. БАЗОВЫЙ ОБРАЗ ----
FROM python:3.10-slim

# ---- 2. СИСТЕМНЫЕ ЗАВИСИМОСТИ ДЛЯ OPENCV ----
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libglib2.0-dev \
    && rm -rf /var/lib/apt/lists/*

# ---- 3. РАБОЧАЯ ДИРЕКТОРИЯ ----
WORKDIR /app

# ---- 4. КОПИРУЕМ ФАЙЛЫ ----
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# ---- 5. ЗАПУСК ----
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
