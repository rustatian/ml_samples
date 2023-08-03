FROM python:3.11

RUN apt update
RUN apt install ffmpeg -y
WORKDIR /app
COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt
COPY ./ /app

CMD ["python", "main.py"]
