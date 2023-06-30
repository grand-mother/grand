FROM fedo_root_35

WORKDIR /opt/grandlib

RUN yum install -y python3-tkinter\
&& yum install -y vim\
&& yum install -y nano\
&& yum install -y gedit

# install quality tools
COPY requirements_qual.txt /opt/grandlib/requirements_qual.txt
RUN python3 -m pip install --no-cache-dir -r /opt/grandlib/requirements_qual.txt

# install documenation tools
COPY requirements_docs.txt /opt/grandlib/requirements_docs.txt
RUN python3 -m pip install --no-cache-dir -r /opt/grandlib/requirements_docs.txt

# other tools for dev
RUN python3 -m pip install --no-cache-dir ipython\
&& python3 -m pip install --no-cache-dir jupyterlab ipympl\
&& python3 -m pip install --no-cache-dir ipynb\
&& echo 'alias grand_jupyter="jupyter-lab --allow-root --ip 0.0.0.0 --no-browser"' >> ~/.bashrc
EXPOSE 8888

WORKDIR /home
