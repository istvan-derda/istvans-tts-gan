FROM docker.io/python:3.11

RUN apt update && apt install -y libgl1-mesa-glx
COPY guessed_requirements.txt /
RUN pip install -r guessed_requirements.txt