# our R base image
FROM rocker/tidyverse

MAINTAINER Youngser Park <youngser@jhu.edu>

# create an R user
ENV HOME /home/user
RUN useradd --create-home --home-dir $HOME user \
    && chown -R user:user $HOME

# install required unix apps
RUN apt-get update && apt-get install -y openssh-server \
    git libcurl4-openssl-dev libxml2-dev \
    libssl-dev libssh2-1-dev vim wget tmux \
    build-essential libreadline-dev python3 python3-dev \
    liblzma-dev bzip2 libbz2-dev libicu-dev

# Pull project
# tocken: gUNY-zMxguoAMW7Qn41t
#RUN git clone https://oauth2:gUNY-zMxguoAMW7Qn41t@gitlab.datadrivendiscovery.org/ypark/D3M.git /home/user/D3M
RUN git clone https://github.com/youngser/D3M.git /home/user/D3M

# install required R packages
RUN echo "r <- getOption('repos'); r['CRAN'] <- 'http://cran.us.r-project.org'; options(repos = r);" > ~/.Rprofile
RUN set -ex \
    && Rscript -e 'install.packages(c("devtools","mclust","popbio","clue", "igraph"), dependencies = c("Depends", "Imports"))' \
    && Rscript -e 'devtools::install_github("youngser/gmmase")' \
    && Rscript -e 'install.packages("http://www.cis.jhu.edu/~parky/D3M/VN_0.3.0.tar.gz",type="source", dependencies = c("Depends", "Imports"))'

# copy required files/primitives
# COPY . $HOME

RUN wget https://bootstrap.pypa.io/get-pip.py && python3 get-pip.py
RUN pip3 install numpy scipy python-igraph rpy2 sklearn jinja2

#RUN mkdir /home/user/D3M/DATA
RUN chown -R user:user /home/user/D3M
RUN chmod -R o+rw /usr/local
RUN echo "/usr/local/lib/R/lib/" > /etc/ld.so.conf.d/libR.conf \
    && cd /etc/ld.so.conf.d/ && ldconfig


WORKDIR $HOME
USER user

# to push to D3M gitlab
# docker login gitlab.datadrivendiscovery.org:5005
# Error response from daemon: Get https://gitlab.datadrivendiscovery.org:5005/v1/users/: dial tcp 192.31.133.81:5005: i/o timeout
# docker build -t gitlab.datadrivendiscovery.org:5005/ypark/d3m/image .
# docker push gitlab.datadrivendiscovery.org:5005/ypark/d3m/image
ENTRYPOINT ["bash"]
