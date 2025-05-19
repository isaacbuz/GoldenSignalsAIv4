FROM python:3.10-slim
WORKDIR /app
COPY presentation/frontend/ .
RUN pip install dash plotly pandas
CMD ["python", "app/dashboard.py"]
