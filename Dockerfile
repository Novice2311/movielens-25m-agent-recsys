FROM python:3.12.10-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
WORKDIR /app
COPY requirements.txt .
COPY wheelhouse /wheelhouse
RUN if ls /wheelhouse/*.whl >/dev/null 2>&1; then pip install --no-index --find-links=/wheelhouse -r requirements.txt; else pip install --timeout 180 --retries 8 -r requirements.txt; fi \
    && useradd --uid 10001 --create-home agent
COPY agent agent
COPY prompts prompts
COPY skills skills
COPY web web
COPY results results
COPY app.py .
COPY llm_proxy.py .
USER 10001:10001
EXPOSE 8080
CMD ["uvicorn","app:app","--host","0.0.0.0","--port","8080"]
