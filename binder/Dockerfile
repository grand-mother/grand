FROM rootproject/root:6.26.02-ubuntu20.04

RUN mkdir -p /opt/grandlib

# Add package for C compilation 
#RUN apt-get update && apt -y upgrade\
RUN apt-get update\
&& apt-get install -y git\
&& apt install -y make\
&& apt install -y libpng-dev\
&& apt install -y pip


# install python lib for grand lib
COPY requirements.txt /opt/grandlib/requirements_grandlib.txt
RUN python3 -m pip install --upgrade pip\
&& python3 -m pip install --no-cache-dir -r /opt/grandlib/requirements_grandlib.txt

# install the notebook package
RUN pip install --no-cache notebook jupyterlab

# create user with a home directory
ARG NB_USER
ARG NB_UID
ENV USER ${NB_USER}
ENV HOME /home/${NB_USER}

RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${NB_UID} \
    ${NB_USER}
WORKDIR ${HOME}
USER ${USER}