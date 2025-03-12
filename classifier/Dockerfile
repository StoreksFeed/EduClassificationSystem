# Use the official Tensorflow image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /code

# Install dependencies
COPY requirements.txt /code/
RUN pip install --upgrade pip
RUN pip install cassandra-driver
RUN pip install torch==2.6.0+cpu --index-url https://download.pytorch.org/whl/cpu
RUN pip install -r /code/requirements.txt

# Copy project
COPY . /code/
