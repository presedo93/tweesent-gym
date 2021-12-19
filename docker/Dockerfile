FROM python:3.9-slim-buster

LABEL com.nvidia.volumes.needed=nvidia_driver

RUN useradd -ms /bin/bash developer

USER developer

WORKDIR /home/developer/plmodel

ENV PATH="/home/developer/.local/bin:${PATH}"

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

CMD [ "python", "-m", "dashboard" ]
