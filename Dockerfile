FROM ubuntu:latest

#
# Essentials: developer tools, build tools
#
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl vim unzip openssh-client wget \
    build-essential cmake checkinstall gcc
#
# Python 3
#
# For convenience, alias (but don't sym-link) python & pip to python3 & pip3 as recommended in:
# http://askubuntu.com/questions/351318/changing-symlink-python-to-python3-causes-problems
RUN apt-get install -y --no-install-recommends python python3-pip python-dev python3-dev && \
    pip3 install --no-cache-dir --upgrade pip setuptools && \
    echo "alias python='python3'" >> /root/.bash_aliases && \
    echo "alias pip='pip3'" >> /root/.bash_aliases

#
# Science libraries and other common packages
#
RUN pip3 --no-cache-dir install \
    numpy scipy==1.1.0 sklearn scikit-image pandas matplotlib Cython requests

COPY requirements.txt /tmp/requirements.txt
RUN pip3 install -U -r /tmp/requirements.txt

#set a directory for the app
WORKDIR /usr/src/deepsnake/

#copy all the files to the container
COPY . .


RUN ROOT=/usr/src/deepsnake 
#RUN cd data && ln -s /data kitti
