FROM nvidia/cuda:12.0-devel-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc g++ python3 python3-pip make cmake git \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -s /bin/bash coderunner
RUN mkdir -p /home/coderunner/workspace
RUN chown -R coderunner:coderunner /home/coderunner

# Set resource limits
RUN echo "coderunner soft nproc 50" >> /etc/security/limits.conf
RUN echo "coderunner hard nproc 100" >> /etc/security/limits.conf
RUN echo "coderunner soft nofile 1024" >> /etc/security/limits.conf
RUN echo "coderunner hard nofile 2048" >> /etc/security/limits.conf

USER coderunner
WORKDIR /home/coderunner/workspace

ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

CMD ["/bin/bash"]