#!/bin/bash
sh -c 'echo "deb http://cran.rstudio.com/bin/linux/ubuntu artful/" >> /etc/apt/sources.list'
gpg --keyserver keyserver.ubuntu.com --recv-key E084DAB9
gpg -a --export E084DAB9 | apt-key add -
sudo apt-get update
sudo apt-get -y install r-base
sudo apt-get -y install r-base-dev
sudo apt-get -y install r-recommended

