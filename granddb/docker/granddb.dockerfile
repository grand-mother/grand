FROM postgres

RUN apt-get update
RUN apt-get -y install ruby ruby-dev libpq-dev build-essential wget
RUN gem install pgsync
RUN curl https://www.pgadmin.org/static/packages_pgadmin_org.pub | apt-key add
RUN sh -c 'echo "deb https://ftp.postgresql.org/pub/pgadmin/pgadmin4/apt/$(lsb_release -cs) pgadmin4 main" > /etc/apt/sources.list.d/pgadmin4.list'
RUN apt-get update
RUN mkdir /app
WORKDIR /app
COPY create-db.bash /app/create-db.bash
COPY pgsync.yml /app/.pgsync.yml
RUN chmod u+x /app/create-db.bash
RUN wget https://github.com/sosedoff/pgweb/releases/download/v0.13.1/pgweb_linux_amd64.zip
RUN unzip pgweb_linux_amd64.zip
REM ./pgweb_linux_amd64 --host localhost --user postgres --db granddb