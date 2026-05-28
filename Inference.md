# Inference — Running the GR00T VLA Server (DGX Spark)

This document describes how to start the GR00T policy as an inference server on
the **DGX Spark** host (`spark-adde`), using the SONIC fine-tune of the
`UNITREE_G1_SONIC` embodiment. The server exposes the policy over the
`PolicyServer` ZMQ protocol so a client (real-robot driver, sim runner, etc.)
can call `get_action(obs)` remotely.

---

## 1. Prerequisites

- Repo cloned at `~/repos/Isaac-GR00T`
- Spark venv already created at `.venv/` (Python 3.12, CUDA 13.0)
- Model checkpoint cached locally — for SONIC WBC-16:
  ```
  /home/microagi/.cache/huggingface/hub/models--LucaFrat--groot-wbc-16/snapshots/5963c8fb0920de2ccb0b5541fbf995b310d84460/groot-wbc-16/checkpoint-8000
  ```
  (Pulled from HF repo `LucaFrat/groot-wbc-16`.)

If the venv or deps are missing, see `scripts/deployment/spark/install_deps.sh`
and `CLAUDE.md` for the full bring-up.

---

## 2. Activate the environment

Open a fresh shell on the Spark host and run the two activation steps **in
order, in the same shell**:

```bash
cd ~/repos/Isaac-GR00T

# 2a. Activate the Python venv (uv-managed)
source .venv/bin/activate

# 2b. Configure Spark-specific CUDA / library paths
source scripts/activate_spark.sh
```

`scripts/activate_spark.sh` sets the variables that PyTorch, Triton, and
torchcodec need on this platform:

- `TRITON_PTXAS_PATH` → `/usr/local/cuda-13.0/bin/ptxas` (needed by `torch.compile`)
- `CUDA_HOME` / `CUDA_PATH` → `/usr/local/cuda-13.0`
- `LD_LIBRARY_PATH` → prepends `torch/lib` and the `nvidia/*/lib` dirs from
  site-packages (cuDNN, cuDSS, etc.)
- `CPATH` / `C_INCLUDE_PATH` / `CPLUS_INCLUDE_PATH` → `${CUDA_HOME}/include`
  (so Triton can find `cuda.h`)

It prints a confirmation line on success:

```
Spark environment configured.
  TRITON_PTXAS_PATH=/usr/local/cuda-13.0/bin/ptxas
  CUDA_HOME=/usr/local/cuda-13.0
  LD_LIBRARY_PATH=...
```

> **Docker note:** If you are running inside the Spark Docker image
> (`scripts/deployment/spark/Dockerfile`), you do **not** need to source
> `activate_spark.sh` — the Dockerfile sets these via `ENV`.

---

## 3. Launch the inference server

With the env active, start the server:

```bash
python -u gr00t/eval/run_gr00t_server.py \
    --model-path /home/microagi/.cache/huggingface/hub/models--LucaFrat--groot-wbc-16/snapshots/5963c8fb0920de2ccb0b5541fbf995b310d84460/groot-wbc-16/checkpoint-8000 \
    --embodiment-tag UNITREE_G1_SONIC \
    --device cuda:0 \
    --port 5550
```

Flags:

| Flag | Value | Meaning |
|---|---|---|
| `-u` | — | Unbuffered Python stdout (so logs stream live in tmux / journald) |
| `--model-path` | checkpoint dir | Path to the fine-tuned checkpoint to load into `Gr00tPolicy` |
| `--embodiment-tag` | `UNITREE_G1_SONIC` | Resolved by `EmbodimentTag.resolve()` — case-insensitive, accepts name or value (`unitree_g1_sonic`) |
| `--device` | `cuda:0` | First Spark GPU |
| `--port` | `5550` | ZMQ port the `PolicyServer` binds to (default in code is `5555`) |

Host defaults to `0.0.0.0`, so the server is reachable from any interface on
the Spark box.

### Expected startup output

```
Starting GR00T inference server...
  Embodiment tag: EmbodimentTag.UNITREE_G1_SONIC
  Model path: /home/microagi/.cache/huggingface/hub/models--LucaFrat--groot-wbc-16/snapshots/5963c8fb0920de2ccb0b5541fbf995b310d84460/groot-wbc-16/checkpoint-8000
  Device: cuda:0
  Host: 0.0.0.0
  Port: 5550
...
✓ Server ready — listening on 0.0.0.0:5550
```

Once you see `Server ready`, the policy is loaded and the socket is bound —
clients can connect.

---

## 4. One-shot copy-paste (full sequence)

```bash
cd ~/repos/Isaac-GR00T
source .venv/bin/activate
source scripts/activate_spark.sh
python -u gr00t/eval/run_gr00t_server.py \
    --model-path /home/microagi/.cache/huggingface/hub/models--LucaFrat--groot-wbc-16/snapshots/5963c8fb0920de2ccb0b5541fbf995b310d84460/groot-wbc-16/checkpoint-8000 \
    --embodiment-tag UNITREE_G1_SONIC \
    --device cuda:0 \
    --port 5550
```

---

## 5. Shutting down

`Ctrl-C` in the server shell → the `KeyboardInterrupt` handler prints
`Shutting down server...` and exits cleanly.

If the server was backgrounded (tmux / nohup), find the PID with
`pgrep -f run_gr00t_server.py` and `kill <pid>`.

---

## 6. Troubleshooting

- **`Model path ... does not exist`** — the checkpoint snapshot hash changed
  after a re-pull. Re-`ls` the `models--LucaFrat--groot-wbc-16/snapshots/`
  directory and update the path.
- **`ptxas not found`** — `activate_spark.sh` printed a warning. Install the
  CUDA 13.0 toolkit or set `TRITON_PTXAS_PATH` manually before launching.
- **`Address already in use` on port 5550** — another server is still bound;
  `pgrep -f run_gr00t_server.py` then `kill`, or pick a different `--port`.
- **Embodiment tag rejected** — pass the exact name `UNITREE_G1_SONIC` or
  value `unitree_g1_sonic`; see `gr00t/data/embodiment_tags.py` for the
  registry.

---

## 7. Related docs

- `CLAUDE.md` — repo-level overview and Spark deployment pointers
- `SONIC_DEPLOY_RUNBOOK.md` — full SONIC bring-up runbook
- `scripts/deployment/spark/` — Spark install scripts and Dockerfile
- `gr00t/eval/run_gr00t_server.py` — server entry point and config schema
