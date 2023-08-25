# get slim base image for python
FROM python:3.9.17-slim-bullseye as builder
# Install OS dependencies
RUN apt-get -y update && apt-get install -y --no-install-recommends \
         ca-certificates \
         dos2unix \
    && rm -rf /var/lib/apt/lists/*
# copy requirements file and and install
COPY ./requirements/requirements.txt /opt/
RUN pip3 install --no-cache-dir -r /opt/requirements.txt
# copy src code into image and chmod scripts
COPY src ./opt/src
COPY ./entry_point.sh /opt/
RUN chmod +x /opt/entry_point.sh
COPY ./fix_line_endings.sh /opt/
RUN chmod +x /opt/fix_line_endings.sh
RUN /opt/fix_line_endings.sh "/opt/src"
RUN /opt/fix_line_endings.sh "/opt/entry_point.sh"
# Set working directory
WORKDIR /opt/src
# set python variables and path
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/src:${PATH}"
# set non-root user
USER 1000
# set entrypoint
ENTRYPOINT ["/opt/entry_point.sh"]
