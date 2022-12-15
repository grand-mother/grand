FROM rootproject/root:6.26.02-ubuntu20.04

RUN mkdir -p /opt/grandlib

# Add package for C compilation 
#RUN apt-get update && apt -y upgrade\
RUN apt-get update\
&& apt-get install -y git\
&& apt install -y make\
&& apt install -y libpng-dev\
&& apt install -y pip


# python is python3
# with 6.26.02-ubuntu20.04
# ln: failed to create symbolic link '/usr/bin/python': File exists
#RUN ln -s /usr/bin/python3 /usr/bin/python


# install python lib for grand lib
COPY requirements.txt /opt/grandlib/requirements_grandlib.txt
RUN python3 -m pip install --upgrade pip\
&& python3 -m pip install --upgrade pip\
&& python3 -m pip install --no-cache-dir -r /opt/grandlib/requirements_grandlib.txt


# init env docker
WORKDIR /home
ENV PATH=".:${PATH}"
ENTRYPOINT ["/bin/bash"]
SHELL ["/bin/bash", "-c"]
