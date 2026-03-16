FROM python:3.11

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

ENV TORCH_HOME=/torch_cache
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

WORKDIR /workspace

# Upgrade pip FIRST
RUN pip install --upgrade pip

COPY requirements.txt .

RUN pip install \
    --no-cache-dir \
    --timeout 600 \
    --retries 10 \
    -r requirements.txt

CMD ["bash"]

