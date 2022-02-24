FROM grandlib_base

WORKDIR /opt/grandlib

RUN apt-get update\
&& apt install -y python3-tk\
&& apt install -y doxygen\
&& apt install -y vim\
&& apt install -y nano

# install quality tools
COPY requirements_qual.txt /opt/grandlib/requirements_qual.txt
RUN python3 -m pip install --no-cache-dir -r /opt/grandlib/requirements_qual.txt

# install documenation tools
COPY requirements_docs.txt /opt/grandlib/requirements_docs.txt
RUN python3 -m pip install --no-cache-dir -r /opt/grandlib/requirements_docs.txt

# other python tools
RUN python3 -m pip install --no-cache-dir ipython

WORKDIR /home
