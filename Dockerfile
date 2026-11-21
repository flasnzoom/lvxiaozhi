# syntax=docker/dockerfile:1
FROM python:3.12-slim


ENV PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app


COPY requirements.txt .
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt


COPY . .


RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser


EXPOSE 8000
CMD ["python", "-m", "uvicorn", "api.api_server:app", "--host", "0.0.0.0", "--port", "8000"]
