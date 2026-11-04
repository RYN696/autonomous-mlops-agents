FROM python:3.11-slim
WORKDIR /app
COPY requirements-docker.txt .
RUN pip install --no-cache-dir --default-timeout=120 --retries 10 -r requirements-docker.txt
# Client Docker + compose : l'agent Fiabilité redémarre / recrée les conteneurs (socket Docker monté)
COPY --from=docker:27.3.1-cli /usr/local/bin/docker /usr/local/bin/docker
COPY --from=docker/compose-bin:v2.29.7 /docker-compose /usr/local/lib/docker/cli-plugins/docker-compose
COPY common ./common
COPY pipeline ./pipeline
COPY agents ./agents
COPY rules ./rules
ENV PYTHONPATH=/app
