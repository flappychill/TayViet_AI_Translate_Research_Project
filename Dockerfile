FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY . /app
ENV PORT=8000
CMD ["gunicorn","-w","2","-b","0.0.0.0:8000","api.app:app"]
