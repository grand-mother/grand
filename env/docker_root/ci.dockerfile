FROM grandlib_base

WORKDIR /home/install

# install sonar-scanner for SonarQub server
RUN apt-get install -y unzip
RUN curl -O https://binaries.sonarsource.com/Distribution/sonar-scanner-cli/sonar-scanner-cli-4.6.2.2472-linux.zip
RUN unzip sonar-scanner-cli-4.6.2.2472-linux.zip && rm sonar-scanner-cli-4.6.2.2472-linux.zip
ENV PATH=$PATH:/home/install/sonar-scanner-4.6.2.2472-linux/bin


# install quality tools
COPY requirements_qual.txt /home/install/requirements_qual.txt
RUN python3 -m pip install --no-cache-dir -r /home/install/requirements_qual.txt

WORKDIR /home
