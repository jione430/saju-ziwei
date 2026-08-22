FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
# 워커 여러 개로 동시성 확보 (pythonmonkey는 스레드가 아닌 프로세스 단위로 격리해야 함 - README 참고)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
