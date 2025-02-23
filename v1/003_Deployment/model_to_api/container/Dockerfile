# Build an image that can do training and inference in SageMaker
# This is a Python 3 image that uses the nginx, gunicorn, flask stack
# for serving inferences in a stable way.

FROM ubuntu:20.04

MAINTAINER jpbarrett13

# 1. Define the packages required in our environment. 
RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         python3 \
         nginx \
         ca-certificates \
         python3-distutils \
         python3-apt \
    && rm -rf /var/lib/apt/lists/*

# 2. Here we define all python packages we want to include in our environment.
# Pip leaves the install caches populated which uses a significant amount of space. 
# These optimizations save a fair amount of space in the image, which reduces start up time.
RUN wget https://bootstrap.pypa.io/get-pip.py && python3 get-pip.py && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir tensorflow==2.13.0 transformers==4.35.2 sentence-transformers==2.2.2 huggingface_hub==0.25.2 pandas numpy flask gevent gunicorn && \
    rm -rf /root/.cache

# 3. Set some environment variables. PYTHONUNBUFFERED keeps Python from buffering our standard
# output stream, which means that logs can be delivered to the user quickly. PYTHONDONTWRITEBYTECODE
# keeps Python from writing the .pyc files which are unnecessary in this case. We also update
# PATH so that the train and serve programs are found when the container is invoked.
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"
ENV LOG4J_FORMAT_MSG_NO_LOOKUPS=true

# 4. Define the folder where our model files are located

# 5. Define the folder where our inference code is located
COPY topic_classifier /opt/program
WORKDIR /opt/program

ENTRYPOINT ["/bin/python3"]

