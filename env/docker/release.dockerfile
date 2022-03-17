FROM grandlib_dev

RUN mkdir -p /opt/grandlib/grand

COPY grand /opt/grandlib/grand
WORKDIR /opt/grandlib/grand
RUN env/setup.sh
RUN pytest tests

ENV PYTHONPATH="/opt/grandlib/grand:${PYTHONPATH}"

WORKDIR /home