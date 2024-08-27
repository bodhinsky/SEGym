FROM python:3.11-alpine
RUN apk update && apk add --no-cache \
    git \
    nano \
    build-base \
    gcc \
    g++ \
    libffi-dev \
    musl-dev \
    linux-headers \
    python3-dev
RUN git clone https://github.com/bodhinsky/apicurl.git repo
WORKDIR repo
RUN git checkout b84fc67c113d76588c80406cb4b3b54b52b16f21
RUN pip install poetry
RUN poetry install
RUN pip install pytest python-dotenv seaborn