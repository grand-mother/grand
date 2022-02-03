FROM grandlib_dev

RUN apt-get update\
&& apt install -y default-jre\
&& apt install -y libswt-gtk-4-java

ENV PATH="${PATH}:/home/install/eclipse"

##&& apt install -y vim\
##&& apt install -y nano

