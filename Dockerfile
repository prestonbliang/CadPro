FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CADPRO_STORAGE_DIR=/var/lib/cadpro/jobs

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN python -m pip install --no-cache-dir . \
    && mkdir -p /var/lib/cadpro/jobs \
    && useradd --create-home --uid 10001 cadpro \
    && chown -R cadpro:cadpro /var/lib/cadpro

USER cadpro
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4)"

CMD ["uvicorn", "cadpro.web:app", "--host", "0.0.0.0", "--port", "8000", "--no-proxy-headers"]
