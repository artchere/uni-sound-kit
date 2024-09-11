FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get upgrade -y && apt-get autoremove && apt-get autoclean
RUN apt-get install -y apt-utils python3-pip wget git ffmpeg libportaudio2 portaudio19-dev alsa-utils alsa-base

ARG PROJECT=usk
ARG PROJECT_DIR=/${PROJECT}
RUN mkdir -p $PROJECT_DIR
WORKDIR $PROJECT_DIR

RUN pip3 install --upgrade pip==24.2

COPY requirements.txt .
RUN pip3 install -r requirements.txt

RUN apt-get install -y locales && locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

WORKDIR $PROJECT_DIR
