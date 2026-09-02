FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY services/ui/ services/ui/

CMD ["streamlit", "run", "services/ui/app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
