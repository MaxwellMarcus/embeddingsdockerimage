FROM python:3.10.0-buster
COPY ./workspace/CT-CLIP/ /CT-CLIP

ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /CT-CLIP

RUN apt-get update -y && apt-get install libgl1 -y

RUN python3.10 -u -m pip install -U pymilvus
RUN python3.10 -u -m pip install -e /CT-CLIP/CT_CLIP
RUN python3.10 -u -m pip install -e /CT-CLIP/transformer_maskgit
#RUN python3.10 -u -m pip install -U numpy
RUN python3.10 -u -m pip install -U transformers==4.23.1

ENV LD_PRELOAD=/usr/local/lib/python3.10/site-packages/sklearn/utils/../../scikit_learn.libs/libgomp-d22c30c5.so.1.0.0

RUN python3 -u scripts/preload.py PROJECT_ID SUBJECT_ID SESSION_ID SCAN_ID test

