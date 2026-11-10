# UDI 2.0 - multi-stage build: React frontend (Node) + Flask API (Python) served by waitress on port 5000.
FROM node:22-slim AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UDI_DATA_DIR=/data MPLBACKEND=Agg
WORKDIR /app/backend
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY backend/ ./
COPY --from=web /web/dist /app/frontend/dist
RUN rm -f .env && useradd -m udi && mkdir -p /data && chown -R udi /data /app
USER udi
VOLUME ["/data"]
EXPOSE 5000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:5000/api/health')" || exit 1
CMD ["python", "serve.py"]
