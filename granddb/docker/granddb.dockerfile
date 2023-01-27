FROM postgres:15.1-bullseye

RUN apt-get update
RUN apt-get -y install ruby ruby-dev libpq-dev build-essential wget curl
RUN gem install pgsync
RUN apt-get update
RUN mkdir /app
WORKDIR /app
COPY create-db.bash /app/create-db.bash
COPY start-web.bash /app/start-web.bash
RUN chmod ugo+rx /app/create-db.bash
RUN chmod ugo+rx /app/start-web.bash
RUN ln -s /app/create-db.bash /app/update-db.bash
COPY pgsync.yml /app/.pgsync.yml
RUN wget https://github.com/sosedoff/pgweb/releases/download/v0.13.1/pgweb_linux_amd64.zip
RUN unzip pgweb_linux_amd64.zip
RUN echo "sed -i -e \"s/trust/scram-sha-256/g\" /var/lib/postgresql/data/pg_hba.conf" > /docker-entrypoint-initdb.d/unset_trust.sh
RUN chmod ugo+rx /docker-entrypoint-initdb.d/unset_trust.sh
EXPOSE 8081
