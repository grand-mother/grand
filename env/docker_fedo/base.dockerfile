FROM fedora:35

RUN mkdir -p /opt/grandlib

# python 3.9
#RUN yum install -y python3.9 \
#&& rm -f /usr/bin/python3 \
#&& ln -s /usr/bin/python3.9 /usr/bin/python3

 

# Add package for C compilation 
#RUN yum -y update\
RUN yum install -y root \
&& yum install -y python3-root\
&& yum install -y git\
&& yum install -y make\
&& yum install -y libpng-devel\
&& yum install -y python3-devel\
&& yum install -y pip


# install python lib for grand lib
COPY requirements.txt /opt/grandlib/requirements_grandlib.txt
RUN python3 -m pip install --upgrade pip\
&& python3 -m pip install --no-cache-dir -r /opt/grandlib/requirements_grandlib.txt


# init env docker
WORKDIR /home
ENV PATH=".:${PATH}"
ENTRYPOINT ["/bin/bash"]
SHELL ["/bin/bash", "-c"]
