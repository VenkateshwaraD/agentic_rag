FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY pyproject.toml README.md ./
RUN pip install streamlit httpx python-dotenv

COPY src/ui ./src/ui
COPY db/seed/users.json ./db/seed/users.json

RUN useradd -r -u 10001 ui && chown -R ui /app
USER ui

EXPOSE 8501
CMD ["streamlit", "run", "src/ui/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
