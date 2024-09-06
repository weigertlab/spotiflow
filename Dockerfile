FROM mambaorg/micromamba:lunar
LABEL authors="Albert Dominguez, Miguel Ibarra"

# Set the base layer for micromamba
USER root
COPY docker-env-config.yml .

RUN apt-get update -qq && apt-get install -y \
    build-essential \
    ffmpeg \
    libsm6 \
    libxext6 \
    procps \
    git

# Set the environment variable for the root prefix
ARG MAMBA_ROOT_PREFIX=/opt/conda

# Add /opt/conda/bin to the PATH
ENV PATH $MAMBA_ROOT_PREFIX/bin:$PATH

# Install stuff with micromamba
RUN micromamba env create -f docker-env-config.yml --yes && \
    micromamba clean --all --yes

# Add environment to PATH
ENV PATH="/opt/conda/envs/spotiflow/bin:$PATH"

# Set the working directory
WORKDIR /spotiflow

# Copy contents of the folder to the working directory
COPY . .