# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.10-slim

EXPOSE 8000

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1
RUN pip install --upgrade pip setuptools wheel
# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1
# Cài đặt thư viện phụ thuộc cho MySQL
RUN apt-get update && apt-get install -y default-libmysqlclient-dev build-essential
RUN pip install --upgrade pip
# Install pip requirements
COPY requirements.txt .
# Cài đặt thư viện cần thiết cho MySQL
RUN apt-get update && apt-get install -y \
    default-libmysqlclient-dev \
    build-essential \
    pkg-config \
    python3-dev
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /app
COPY . /app

# Creates a non-root user with an explicit UID and adds permission to access the /app folder
# For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app
USER appuser

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "Drink_Marketplace.wsgi"]
