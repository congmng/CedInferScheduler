# Router image: the vLLM image already carries fastapi/uvicorn/httpx, but the
# shared CASR LP backend needs OR-Tools and the +P/-P lifecycle needs an ssh
# client to start/stop Prefill containers on worker hosts.
FROM vllm/vllm-openai:latest
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install --no-cache-dir -i "$PIP_INDEX_URL" ortools \
 && apt-get update -qq \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends openssh-client \
 && rm -rf /var/lib/apt/lists/*
