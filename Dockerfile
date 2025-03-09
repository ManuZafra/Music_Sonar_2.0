FROM python:3.9-slim
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*
ENV PYTHONUNBUFFERED=1
ENV PATH="/usr/local/bin:${PATH}"
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
CMD ["python", "app.py"]
