FROM python:3.12-slim

WORKDIR /app
COPY backend/requirements.txt .

RUN uv pip install -r requirements.txt --system

RUN python -m crawl4ai.async_webcrawler