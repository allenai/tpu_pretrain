FROM gcr.io/tpu-pytorch/xla:nightly  # Docker image for PyTorch/XLA for the nightly build

RUN apt-get install -y gdb
Run apt-get install -y vim
RUN pip install ipdb
RUN pip install pytorch-transformers

RUN echo "export XRT_TPU_CONFIG=\"tpu_worker;0;\$TPU_IP:8470\"" >> ~/.bashrc