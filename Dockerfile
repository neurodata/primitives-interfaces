FROM registry.datadrivendiscovery.org/jpl/docker_images/complete:ubuntu-artful-python36-v2018.7.10

RUN mkdir -p /user_opt
RUN mkdir -p /output
RUN mkdir -p /input

COPY ta1-pipeline /user_opt
COPY startup.sh /user_opt
COPY 85153fef-de46-43af-8a6c-00bab0c05cb5.json /user_opt

RUN apt-get update && apt-get install -y openssh-server \
    git libcurl4-openssl-dev libxml2-dev \
    libssl-dev libssh2-1-dev vim wget tmux \
    build-essential libreadline-dev python3.6 python3.6-dev \
    liblzma-dev bzip2 libbz2-dev libicu-dev 

RUN apt-get update && apt-get -y install gfortran libblas-dev liblapack-dev libjpeg-dev

RUN wget http://ftp.us.debian.org/debian/pool/main/c/cdbs/cdbs_0.4.156_all.deb
RUN dpkg -i cdbs_0.4.156_all.deb

# RUN wget http://launchpadlibrarian.net/340609990/r-base-dev_3.4.2-1ubuntu1_all.deb
# RUN dpkg -i r-base-dev_3.4.2-1ubuntu1_all.deb

RUN apt-get update
# RUN apt-get install r-base
# RUN apt-get install r-base-dev
# RUN echo "r <- getOption('repos'); r['CRAN'] <- 'http://cran.us.r-project.org'; options(repos = r);" > ~/.Rprofile
RUN chmod a+x /user_opt/startup.sh

CMD ["/user_opt/startup.sh"]
