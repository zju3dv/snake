FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

#
# Essentials: developer tools, build tools
#
RUN apt-get update && apt-get install -y --no-install-recommends --fix-missing \
    git curl vim unzip openssh-client wget \
    build-essential cmake checkinstall gcc tmux \
#Other requirements
    libgtk2.0-dev
#
# Python 3
#
# For convenience, alias (but don't sym-link) python & pip to python3 & pip3 as recommended in:
# http://askubuntu.com/questions/351318/changing-symlink-python-to-python3-causes-problems
RUN apt-get install -y --no-install-recommends python python3-pip python-dev python3-dev && \
    pip3 install --no-cache-dir --upgrade pip setuptools && \
    echo "alias python='python3'"   >> /root/.bash_aliases && \
    echo "alias pip='pip3'"         >> /root/.bash_aliases

#
# Science libraries and other common packages
#
RUN pip3 --no-cache-dir install \
    numpy scipy==1.1.0 sklearn scikit-image pandas matplotlib Cython requests \
    cupy==4.1.0 wheel

#set a ROOT directory for the app
WORKDIR /usr/src/deepsnake/

ENV ROOT /usr/src/deepsnake
ENV CUDA_HOME /usr/local/cuda-10.0

#uncomment for use on Turing arch such as Geforce 2070, 2080 etc.
ENV CUDA_LAUNCH_BLOCKING=1

COPY requirements.txt /tmp/requirements.txt
RUN pip3 install -U -r /tmp/requirements.txt

#copy all the files to the container
COPY . .

# install torch 1.1 built from cuda 10.0
#RUN pip3 install torch==1.0.0 -f https://download.pytorch.org/whl/cu100/stable
RUN pip3 install -U https://download.pytorch.org/whl/cu100/torch-1.0.0-cp36-cp36m-linux_x86_64.whl && \
# install apex(lib from nvidia) in the same container
    git clone https://github.com/DesperateMaker/apex.git /usr/src/apex && cd /usr/src/apex/ # && \
#    python3 setup.py install --cuda_ext --cpp_ext && \
# install extentions
#    cd ${ROOT}/lib/csrc/dcn_v2          && python3 setup.py build_ext --inplace && \
#    cd ${ROOT}/lib/csrc/extreme_utils   && python3 setup.py build_ext --inplace && \
#    cd ${ROOT}/lib/csrc/roi_align_layer && python3 setup.py build_ext --inplace

# RUN cd ${ROOT}/data && ln -s /data kitti