FROM tensorflow/tensorflow:latest-gpu-jupyter
RUN apt-get upate && apt-get install -y \
ca-certificates \
curl \
git \
libfreetype6-dev \
python3-dev \
python3-pip \
vim \
wget

RUN pip3 install --upgrate pip
RUN pip3 install tensorflow-gpu
RUN pip3 install h5py jupyter matplotlib numpy pandas pyyaml seaborn sklearn
RUN pip3 install keras --no-deps
RUN pip3 install keras_applications --no-deps
RUN pip3 install keras_preprocessing --no-deps
RUN pip3 install statsmodels
RUN pip3 install tqdm
RUN pip3 install tabulate
RUN ["mkdir", "notebooks"]
COPY jupyter_notebook_config.py /root/.jupyter/
COPY run_jupyter.sh /

EXPOSE 8888 6006

VOLUME /notebooks

CMD ["/run_jupyter.sh"]
