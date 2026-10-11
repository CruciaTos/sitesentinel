FROM python:3.12-alpine

ENV PYTHONUNBUFFERED=1
EXPOSE 8080
WORKDIR /app

COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt
COPY . .

CMD ["python3", "app.py"]
