FROM condaforge/mambaforge:latest

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHON_VERSION=3.11
ENV CONDA_ENV_NAME=bagle_env
ENV CONDA_DEFAULT_ENV=bagle_env

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

RUN mamba create -n "${CONDA_ENV_NAME}" python="${PYTHON_VERSION}" pip=24.3.1 -y && \
    mamba clean -afy

ENV PATH=/opt/conda/envs/${CONDA_ENV_NAME}/bin:/opt/conda/condabin:${PATH}

RUN printf '%s\n' \
    'source /opt/conda/etc/profile.d/conda.sh' \
    'conda activate bagle_env' \
    >> /root/.bashrc && \
    cp /root/.bashrc /root/.bash_profile

SHELL ["conda", "run", "--no-capture-output", "-n", "bagle_env", "/bin/bash", "-c"]

RUN mamba install -n "${CONDA_ENV_NAME}" -y \
    ipykernel \
    ipywidgets \
    jupyterlab \
    notebook && \
    mamba clean -afy

RUN pip install \
    astropy==7.1.0 \
    celerite==0.4.3 \
    contourpy==1.3.3 \
    cycler==0.12.1 \
    dynesty==2.1.5 \
    ephem==4.2 \
    graphviz==0.21 \
    joblib==1.5.2 \
    jplephem==2.23 \
    matplotlib==3.10.6 \
    numpy==2.3.3 \
    numpydoc==1.9.0 \
    pandas==2.3.2 \
    pyerfa==2.0.1.5 \
    pytest==8.4.2 \
    scipy==1.16.2 \
    ffmpeg-python 

WORKDIR /opt
RUN git clone --depth 1 --branch master https://github.com/JohannesBuchner/MultiNest.git
WORKDIR /opt/MultiNest/build
RUN cmake -DCMAKE_INSTALL_PREFIX=/opt/MultiNest .. && \
    make && \
    make install

ENV LD_LIBRARY_PATH=/opt/MultiNest/lib

RUN pip install pymultinest==2.12
RUN python -m ipykernel install --name bagle_env --display-name "Python (bagle_env)"

WORKDIR /opt/BAGLE_Microlensing
COPY . /opt/BAGLE_Microlensing
RUN pip install .

RUN mkdir -p /data/input /data/output /data/test

COPY docker/test_imports.py /opt/test_imports.py

WORKDIR /data

EXPOSE 8888
CMD ["/bin/bash"]
