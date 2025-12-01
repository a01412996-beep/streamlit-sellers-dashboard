# 1. Base image with Python
FROM python:3.11-slim

# 2. Working directory inside the container
WORKDIR /app

# 3. Copy dependency list and install
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy the rest of the project (code + CSV)
COPY . /app

# 5. Expose Streamlit's default port
EXPOSE 8501

# 6. Command to start your Streamlit app
CMD ["streamlit", "run", "ML_Model.py", "--server.port=8501", "--server.address=0.0.0.0"]
