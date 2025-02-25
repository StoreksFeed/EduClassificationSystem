# Use the official Python image from the Docker Hub
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set work directory
WORKDIR /code

# Install system dependencies (not sure if they are needed)
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    && apt-get clean

# Install dependencies
COPY requirements.txt /code/
RUN pip install --upgrade pip
RUN pip install -r /code/requirements.txt

# Copy project
COPY . /code/
