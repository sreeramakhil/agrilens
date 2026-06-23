# Use standard lightweight Python base image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app   

# Install system dependencies needed for compiling certain packages or image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to take advantage of Docker caching layers
COPY requirements.txt .

# Install dependencies (use --no-cache-dir to minimize image size)
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application assets and code
COPY data/ ./data/
COPY models/ ./models/
COPY style.css .
COPY streamlit_app.py .

# Expose Streamlit's default port
EXPOSE 8501

# Add standard healthcheck to let deployment platform know container is healthy
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run the application
ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
