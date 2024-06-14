FROM python:3.10.13-slim
RUN apt-get update && apt-get install -y redis-server

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install supervisor

COPY . /app

COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

EXPOSE 5000
#EXPOSE 5555
#EXPOSE 6379  # Port Redis

# Menjalankan supervisord
CMD ["supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
