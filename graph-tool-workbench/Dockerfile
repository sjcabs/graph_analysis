FROM gcr.io/deeplearning-platform-release/workbench-container:latest

# Install Miniforge
RUN wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh && \
    bash /tmp/miniforge.sh -b -p /opt/conda && \
    rm /tmp/miniforge.sh

# Add conda to PATH
ENV PATH="/opt/conda/bin:$PATH"
ENV LD_LIBRARY_PATH="/opt/conda/lib:$LD_LIBRARY_PATH"

# Install all packages via mamba
RUN mamba install -y -c conda-forge \
    python=3.10 \
    graph-tool \
    "networkx>=3.0" \
    "numpy>=1.24" \
    "scipy>=1.10" \
    "matplotlib>=3.7" \
    "scikit-learn<1.8" \
    "graspologic>=3.4,<3.5" \
    gcsfs \
    python-dotenv \
    "pyarrow>=14.0.0" \
    ipykernel \
    ipywidgets \
    git \
    plotly \
    tqdm && \
    mamba clean -afy

# Install ipysigma in conda (kernel)
RUN /opt/conda/bin/pip install ipysigma

# Install ipysigma in JupyterLab 4 environment (frontend)
RUN /opt/micromamba/envs/jupyterlab4/bin/pip install ipysigma

# Clone repo
RUN git clone https://github.com/sjcabs/graph_analysis.git /home/jupyter/sjcabs-graph-analysis

# Register conda kernel
RUN /opt/conda/bin/python -m ipykernel install --name graph-tool --display-name "Python 3.10 (graph-tool)"