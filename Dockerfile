# cds-torch: steps 02-05 (PyTorch). Built by run.sh:  docker build -t cds-torch:2026-11 .
# Base: pytorch/pytorch image, pinned by digest (the environment the paper's models were trained in).
FROM pytorch/pytorch@sha256:82e0d379a5dedd6303c89eda57bcc434c40be11f249ddfadfd5673b84351e806

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt
WORKDIR /work
