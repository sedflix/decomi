# A parent Dockerfile for running the project

#FROM pytorch/pytorch:0.4.1-cuda9-cudnn7-runtime

FROM tensorflow/tensorflow:latest-gpu-py3-jupyter
LABEL maintainer "Siddharth Yadav <sedflix@gmail.com>"

#RUN apt-get update && conda install -y pandas nltk scipy keras matplotlib scikit-learn faiss-gpu -c pytorch && conda clean --packages --yes

RUN apt-get update && pip install pandas nltk scipy keras matplotlib scikit-learn 
CMD ["bash"]
