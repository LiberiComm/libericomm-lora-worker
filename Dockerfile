FROM runpod/serverless:0.4.0-py3.10-cuda11.8

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "-u", "handler.py"]
