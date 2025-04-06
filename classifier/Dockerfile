FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the downloaded model from the builder stage
COPY download.py .
RUN python download.py get_model DeepPavlov/rubert-base-cased
RUN python download.py get_morph

# Copy project
COPY . /app

EXPOSE 8001
CMD ["python", "classifier.py", "runserver", "8001"]
