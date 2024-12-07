FROM python:3.10-slim

WORKDIR /arewetoxic

COPY requirements_app.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY .env .env
COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "scripts/app.py", "--server.port=8501", "--server.address=0.0.0.0"]