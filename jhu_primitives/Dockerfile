# our R base image
FROM ubuntu:16.04

MAINTAINER Youngser Park <youngser@jhu.edu>, Disa Mhembere, Eric Bridgeford

# create an R user
ENV HOME /home/user
RUN useradd --create-home --home-dir $HOME user \
    && chown -R user:user $HOME

# there are snakes on the plane ...
RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository ppa:jonathonf/python-3.6 && \
        add-apt-repository ppa:marutter/rrutter

# install required unix apps
RUN apt-get update && apt-get install -y openssh-server \
    git libcurl4-openssl-dev libxml2-dev \
    libssl-dev libssh2-1-dev vim wget tmux \
    build-essential libreadline-dev python3.6 python3.6-dev \
    liblzma-dev bzip2 libbz2-dev libicu-dev r-base r-base-core

# install required R packages
RUN echo "r <- getOption('repos'); r['CRAN'] <- 'http://cran.us.r-project.org'; options(repos = r);" > ~/.Rprofile
RUN set -ex \
    && Rscript -e 'install.packages(c("devtools","mclust","popbio","clue", "igraph"), dependencies = c("Depends", "Imports"))' \
    && Rscript -e 'devtools::install_github(c("youngser/gmmase", "youngser/VN"))' \
    && Rscript -e 'install.packages("http://www.cis.jhu.edu/~parky/D3M/VN_0.3.0.tar.gz",type="source", dependencies = c("Depends", "Imports"))'

# copy required files/primitives
# COPY . $HOME

# Pull project
RUN git clone https://github.com/neurodata/primitives-interfaces.git \
    /home/user/primitives-interfaces

RUN wget https://bootstrap.pypa.io/get-pip.py && python3.6 get-pip.py
RUN pip3.6 install numpy scipy python-igraph rpy2 sklearn \
        jinja2 matplotlib jhu-primitives jupyter

RUN chmod -R o+rw /usr/local
RUN echo "/usr/local/lib/R/lib/" > /etc/ld.so.conf.d/libR.conf \
    && cd /etc/ld.so.conf.d/ && ldconfig

WORKDIR /home/user/primitives-interfaces/
RUN mkdir -p /home/user/.jupyter
COPY jupyter_notebook_config.py /home/user/.jupyter/jupyter_notebook_config.py
WORKDIR /home/user

EXPOSE 8888
ENTRYPOINT ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
