FROM condaforge/mambaforge:latest

ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_ENV_NAME=bagle_env
ENV CONDA_DEFAULT_ENV=bagle_env
ENV PYTHON_VERSION=3.14

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

RUN echo "Using Python version: ${PYTHON_VERSION}" && \
    mamba create -n "${CONDA_ENV_NAME}" python="${PYTHON_VERSION}" pip -y && \
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
COPY pyproject.toml README.md /opt/BAGLE_Microlensing/
COPY src /opt/BAGLE_Microlensing/src
RUN pip install --upgrade pip setuptools wheel && \
    pip install . pytest jaxns
# jaxns 2.6.x assigns documentation to typing.Union aliases during import.
# Python 3.14 makes those aliases immutable; removing only the documentation
# assignments preserves the runtime types used by jaxns.
RUN sed -i \
    -e '/^Array\.__doc__ =/d' \
    -e '/^FloatArray\.__doc__ =/d' \
    -e '/^IntArray\.__doc__ =/d' \
    -e '/^BoolArray\.__doc__ =/d' \
    "/opt/conda/envs/${CONDA_ENV_NAME}/lib/python${PYTHON_VERSION}/site-packages/jaxns/internals/types.py"
RUN pip install ffmpeg-python graphviz ipykernel ipywidgets jupyterlab
RUN python -m ipykernel install --name bagle_env --display-name "Python (bagle_env)"

RUN mamba install -c conda-forge ffmpeg -y && mamba clean -afy
RUN pip install pyLIMA==1.9.8 VBMicrolensing==5.3.3 MulensModel==3.3.1

COPY docker/test_imports.py /opt/test_imports.py

ENV PYTHONPATH=/opt/BAGLE_Microlensing/src
ENV MPLCONFIGDIR=/tmp/matplotlib

RUN mkdir -p /data/input /data/output /data/test /opt/BAGLE_time_tests "${MPLCONFIGDIR}"

WORKDIR /opt/BAGLE_time_tests

EXPOSE 8888
CMD ["/bin/bash"]
