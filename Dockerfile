# Use an official Python runtime as the base image
FROM python:3.11-slim

# Set the working directory in the container to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

# Specify the command to run on container start
CMD ["gunicorn", "application:application", "--bind", "0.0.0.0:8000"]


# docker build -t wine-quality-prediction .
# docker run -p 8000:8000 wine-quality-prediction
# http://localhost:8000/
