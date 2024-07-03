FROM python:3.10.0-buster
COPY ./workspace/ /workspace

ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /workspace

RUN python3.10 -u -m pip install -U pymilvus
RUN python3.10 -u -m pip install -U pillow
RUN python3.10 -u -m pip install -U pydicom
#RUN python3.10 -u -m pip install -U medclip
RUN python3.10 -u -m pip install -e /workspace/MedCLIP
RUN python3.10 -u -m pip install -U numpy
RUN python3.10 -u -m pip install -U transformers==4.23.1

ENV LD_PRELOAD=/usr/local/lib/python3.10/site-packages/sklearn/utils/../../scikit_learn.libs/libgomp-d22c30c5.so.1.0.0

RUN python3 -u load.py PROJECT_ID SUBJECT_ID SESSION_ID SCAN_ID test

