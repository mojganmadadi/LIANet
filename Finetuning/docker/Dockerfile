FROM pytorch/pytorch:2.3.0-cuda11.8-cudnn8-devel

# System packages + LaTeX + GDAL for rasterio
RUN apt update && apt install -y \
    tmux htop build-essential nano git curl \
    texlive-latex-base dvipng cm-super \
    gdal-bin libgdal-dev \
    && apt clean

# Required for pip-based rasterio installation
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

# Python packages
RUN pip install --upgrade pip && \
    pip install \
        fastprogress jupyter matplotlib tqdm tensorboard \
        scikit-learn scikit-image easydict tensorboardX torchinfo \
        argconfigparse torchmetrics seaborn hydra-core shapely \
        geopandas xarray timm einops h5py rasterio sarpy

ENV workdir=/home/user
WORKDIR ${workdir}

CMD ["bash"]

