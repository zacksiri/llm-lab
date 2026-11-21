FROM vllm/vllm-openai@sha256:c2dc759f5984f69c1e2418c6af7c26eb756f5ed2a8934c768ae1253645d1449e

COPY site/ /sm80site/
COPY patches/vllm/ /usr/local/lib/python3.14/dist-packages/vllm/
COPY overrides/sm86/gemv.py /usr/local/lib/python3.14/dist-packages/vllm/models/inkling2/nvidia/ops/gemv.py
COPY overrides/sm86/vocab_parallel_embedding.py /usr/local/lib/python3.14/dist-packages/vllm/model_executor/layers/vocab_parallel_embedding.py

ENV PYTHONPATH=/sm80site
