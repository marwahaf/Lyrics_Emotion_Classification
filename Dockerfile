# Base image for Python 3.12
FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Copy only the requirements file first
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Volume pour les datasets
VOLUME ["/app/datasets"]

# Copy the source code
COPY . .

# Define the command to run when the container starts
CMD ["python", "main.py"]
