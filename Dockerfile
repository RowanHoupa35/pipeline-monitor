FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Pre-generate logs at build time (optional; can be run at startup too)
# RUN python -m simulator.run_pipeline

EXPOSE 8501

# ANTHROPIC_API_KEY must be passed at runtime via --env or docker-compose
ENV PYTHONUNBUFFERED=1

CMD ["streamlit", "run", "dashboard/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
