FROM condaforge/mambaforge:latest

ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_ENV_NAME=bagle_env
ENV CONDA_DEFAULT_ENV=bagle_env

COPY pyproject.toml /tmp/pyproject.toml

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    gfortran \
    git \
    liblapack-dev \
    libopenblas-dev \
    make \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN PYTHON_VERSION=$(grep 'requires-python' /tmp/pyproject.toml | grep -oE '[0-9]+\.[0-9]+' | head -1) && \
    echo "Using Python version: $PYTHON_VERSION" && \
    mamba create -n "${CONDA_ENV_NAME}" python="${PYTHON_VERSION}" pip=24.3.1 -y && \
    mamba clean -afy

ENV PATH=/opt/conda/envs/${CONDA_ENV_NAME}/bin:/opt/conda/condabin:${PATH}

RUN printf '%s\n' \
    'source /opt/conda/etc/profile.d/conda.sh' \
    'conda activate bagle_env' \
    >> /root/.bashrc && \
    cp /root/.bashrc /root/.bash_profile

SHELL ["conda", "run", "--no-capture-output", "-n", "bagle_env", "/bin/bash", "-c"]

WORKDIR /opt
RUN git clone --depth 1 --branch master https://github.com/JohannesBuchner/MultiNest.git
WORKDIR /opt/MultiNest/build
RUN cmake -DCMAKE_INSTALL_PREFIX=/opt/MultiNest .. && \
    make && \
    make install

ENV LD_LIBRARY_PATH=/opt/MultiNest/lib

WORKDIR /opt/BAGLE_Microlensing
COPY pyproject.toml README.md requirements.txt /opt/BAGLE_Microlensing/
RUN pip install -r requirements.txt
COPY src /opt/BAGLE_Microlensing/src
RUN pip install --no-deps .
RUN pip install ffmpeg-python graphviz ipykernel ipywidgets jupyterlab
RUN python -m ipykernel install --name bagle_env --display-name "Python (bagle_env)"

RUN mamba install -c conda-forge ffmpeg -y && mamba clean -afy
RUN pip install pyLIMA==1.9.8 VBMicrolensing==5.3.3 MulensModel==3.3.1

COPY . /opt/BAGLE_Microlensing

# RUN mkdir -p /data/input /data/output /data/test

COPY docker/test_imports.py /opt/test_imports.py

WORKDIR /data

EXPOSE 8888
CMD ["/bin/bash"]
