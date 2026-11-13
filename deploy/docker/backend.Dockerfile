FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

# CPU-only torch keeps the image sane for the local reranker
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.4.1

COPY pyproject.toml README.md ./
COPY src ./src
COPY db ./db
COPY eval ./eval

RUN pip install -e . \
    && python -m spacy download en_core_web_lg

RUN useradd -r -u 10001 copilot && chown -R copilot /app
USER copilot

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
