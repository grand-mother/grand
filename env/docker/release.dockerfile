FROM grandlib_dev

RUN mkdir -p /home/install/src/grand

COPY grand /home/install/src/grand
WORKDIR /home/install/src/grand
RUN ls -la
RUN env/setup.sh

ENV PYTHONPATH="/home/install/src/grand:${PYTHONPATH}"

WORKDIR /home