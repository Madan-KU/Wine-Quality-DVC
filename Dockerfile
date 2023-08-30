# Use an official Python runtime as the base image
FROM python:3.11-slim

# Set the working directory in the container to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

# For Local
# # Specify the command to run on container start
# CMD ["gunicorn", "application:application", "--bind", "0.0.0.0:8000"]


# EXPOSE 5000
EXPOSE $PORT

#For Heroku
CMD gunicorn --workers=4 --bind 0.0.0.0:$PORT application:application


# docker build -t wine-quality-prediction .
# docker run -p 8000:8000 wine-quality-prediction
# http://localhost:8000/
