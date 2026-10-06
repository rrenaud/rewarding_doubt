# The Modal image (modal_repro/app.py) as a public Docker image for RunPod pods: the released
# RewardingDoubt code at the pinned commit with its pinned requirements, our trainers next to
# Train.py, our package on PYTHONPATH, and runpod/entry.sh as the start command.
# Built and pushed by .github/workflows/docker.yml as ghcr.io/rrenaud/rewarding-doubt.
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends git build-essential \
    && rm -rf /var/lib/apt/lists/*
# The heavy, rarely changing layer first, so code changes rebuild in seconds.
RUN git clone -q https://github.com/pasta99/RewardingDoubt /opt/RewardingDoubt \
    && git -C /opt/RewardingDoubt checkout -q 416c9e0de22a387c35a51260e7739fa28a37865e \
    && pip install --no-cache-dir -r /opt/RewardingDoubt/requirements.txt
# The released Train.py with --objective exact / exact_fast (default ppo unchanged).
COPY patches/exact_confidence.patch /opt/exact_confidence.patch
RUN git -C /opt/RewardingDoubt apply /opt/exact_confidence.patch
COPY modal_repro/subset.py modal_repro/exact_llama.py modal_repro/shared_prefix.py modal_repro/thinking_llama.py \
     modal_repro/frozen_eval.py modal_repro/minimal_trainer.py \
     /opt/RewardingDoubt/SingleAnswerSetting/
COPY src/rewarding_doubt /opt/rd/rewarding_doubt
COPY runpod/entry.sh runpod/selfstop.py runpod/eval_snapshots.py /opt/runpod/
# HF_HOME on the network volume: the model and TriviaQA download once per volume, not once per pod.
ENV PYTHONPATH=/opt/rd HF_HOME=/workspace/hf HF_HUB_ENABLE_HF_TRANSFER=1 TOKENIZERS_PARALLELISM=false \
    PYTHONUNBUFFERED=1 WANDB_MODE=disabled CODE_DIR=/opt/RewardingDoubt/SingleAnswerSetting \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ARG GIT_SHA=unknown
ENV IMAGE_GIT_SHA=$GIT_SHA
CMD ["bash", "/opt/runpod/entry.sh"]
