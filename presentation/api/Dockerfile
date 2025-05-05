FROM python:3.10-slim
WORKDIR /app
COPY presentation/api/requirements.txt .
RUN pip install -r requirements.txt
COPY presentation/api/ .
COPY infrastructure/auth/ infrastructure/auth/
COPY infrastructure/config/ infrastructure/config/
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
