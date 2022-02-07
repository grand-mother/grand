FROM grandlib_dev

RUN mkdir -p /opt/eclipse_install

RUN apt-get update\
&& apt install -y default-jre\
&& apt install -y libswt-gtk-4-java

ENV PATH="${PATH}:/opt/eclipse_install/eclipse"

