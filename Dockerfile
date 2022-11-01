FROM python:3.11.0

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /usr/src/app

# Зависимости скачиваются долго, а меняются реже, чем код, поэтому такой порядок команд
COPY requirements.txt requirements.txt
RUN pip install -r ./requirements.txt
COPY app .
