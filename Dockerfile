FROM runpod/serverless:latest

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "-u", "handler.py"]
