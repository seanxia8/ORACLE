FROM cverstege/alma9-gridjob

WORKDIR /opt/delight-reconstruction

COPY pyproject.toml uv.lock ./

RUN python3.9 -m ensurepip --upgrade || dnf install -y python3.9-pip \
    && python3.9 -m pip install --no-cache-dir uv \
    && uv venv --python python3.9 /opt/delight-venv \
    && UV_PROJECT_ENVIRONMENT=/opt/delight-venv uv sync --frozen --no-dev

ENV PATH="/opt/delight-venv/bin:${PATH}" \
    RECONSTRUCTION_USE_PREBUILT_ENV=1 \
    RECONSTRUCTION_DISABLE_TORCH_COMPILE=1 \
    TORCHDYNAMO_DISABLE=1
