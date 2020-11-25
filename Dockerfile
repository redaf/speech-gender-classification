FROM python

RUN apt update -y
RUN apt install -y lsb-release software-properties-common libsndfile1

WORKDIR /tmp
RUN wget https://apt.llvm.org/llvm.sh
RUN bash llvm.sh 10 && rm llvm.sh

RUN LLVM_CONFIG=/usr/bin/llvm-config-10 pip install librosa pandas mglearn

WORKDIR /home
COPY train.py .
COPY predict.py .
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh
ENTRYPOINT ["/bin/bash", "/home/entrypoint.sh"]