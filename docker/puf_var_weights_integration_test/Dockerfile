# Use Ubuntu 24.04 (Noble Numbat) as the base image
FROM ubuntu:24.04

# Set DEBIAN_FRONTEND to noninteractive to avoid prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Update package lists and install necessary packages
# - git: for cloning repositories
# - any other dependencies your script or code needs (e.g., python3, build-essential, nodejs, etc.)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    openssh-client \
    curl \ 
    python3 \
    python3-pip \
    build-essential \
    libhdf5-dev \
    libblosc-dev \
    liblzo2-dev \
    libbz2-dev \
    zlib1g-dev \
    # And any other tools the script needs
    && \
    curl -LsSf https://astral.sh/uv/install.sh | sh \
    # Clean up and get out
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# NOTE: with the uv install, I may need to run source ~/.bashrc

# Set the working directory inside the container
WORKDIR /app

# Copy your shell script into the container's /app directory
COPY run_test.sh .
COPY check_estimates.py .

# Make the script executable inside the container
RUN chmod +x run_test.sh

# The command that will be run when the container starts
CMD ["./run_test.sh"]
