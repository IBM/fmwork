# CONTEXT {'gpu_vendor': 'AMD', 'guest_os': 'UBUNTU'}
ARG BASE_DOCKER=rocm/vllm-dev:nightly_main_20250420
FROM $BASE_DOCKER

USER root
ENV APP_MOUNT=/app
RUN mkdir -p $APP_MOUNT
WORKDIR $APP_MOUNT

ARG GFX_COMPILATION_ARCH="gfx942"
ENV PYTORCH_ROCM_ARCH=${GFX_COMPILATION_ARCH}

# -----------------------
# hipBLASLt
ARG BUILD_HIPBLASLT="1"
ARG HIPBLASLT_BRANCH="aa0bda7b"
ARG HIPBLASLT_REPO="https://github.com/ROCm/hipBLASLt.git"
RUN if [ "$BUILD_HIPBLASLT" = "1" ]; then \
    cd ${APP_MOUNT} \
    && git clone --recursive ${HIPBLASLT_REPO} \
    && cd hipBLASLt \
    && git checkout ${HIPBLASLT_BRANCH} \
    && ./install.sh -idc -a ${PYTORCH_ROCM_ARCH}; \
    fi

# -----------------------
# vllm
COPY ./vllm_mllama_hip_graph.path /app/
ARG BUILD_VLLM="1"
ARG VLLM_BRANCH="aiter_integration_final"
ARG VLLM_REPO="https://github.com/ROCm/vllm.git"
RUN if [ "$BUILD_VLLM" = "1" ]; then \
    cd ${APP_MOUNT} \
    && pip uninstall -y vllm \
    && rm -rf vllm \
    && git clone --recursive -b ${VLLM_BRANCH} ${VLLM_REPO} \
	  && cd vllm \
    && mv /app/vllm_mllama_hip_graph.path ./  \ 
    && git apply vllm_mllama_hip_graph.path \ 
    && pip install -r requirements/rocm.txt \ 
    && python3 setup.py install; \
    fi
