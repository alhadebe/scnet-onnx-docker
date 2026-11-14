# SCNet ONNX Docker benchmark

Run the **same four-stem SCNet ONNX model** on an AMD/Intel CPU or an NVIDIA CUDA GPU, entirely in Docker. Each run creates `vocals.wav`, `drums.wav`, `bass.wav`, `other.wav`, and `benchmark.json`.

This is a benchmark of full-track separation for potential use in Mixxx's load-time stem preparation—not a Mixxx plugin.

## Requirements

- Docker Engine and Docker Compose v2 on Linux.
- First build and first inference require Internet access to download dependencies and model weights.
- CPU: no special drivers.
- GPU: NVIDIA driver and NVIDIA Container Toolkit configured for Docker, with a driver compatible with CUDA 12.8 (typically version 570+).
- Prefer 8 GB+ available memory, plus storage for four uncompressed stereo WAV files.
- No host-side Python, Node.js, PyTorch, or CUDA toolkit installation is required.

## Quick start

```bash
git clone https://github.com/alhadebe/scnet-onnx-docker.git
cd scnet-onnx-docker
```

Copy the **same audio track** to `audio/test.mp3` on both machines. Any file type supported by FFmpeg can be used.

### AMD CPU

```bash
docker compose --profile cpu run --build --rm cpu /data/test.mp3
```

### NVIDIA CUDA GPU

Check Docker GPU access first:

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi
```

Then run:

```bash
docker compose --profile gpu run --build --rm gpu /data/test.mp3
```

To benchmark the larger model instead:

```bash
MODEL=scnet-large docker compose --profile cpu run --build --rm cpu /data/test.mp3
MODEL=scnet-large docker compose --profile gpu run --build --rm gpu /data/test.mp3
```

## Output

- CPU: `output/test_scnet_cpu/`
- NVIDIA: `output/test_scnet_cuda/`
- Large models use `scnet-large` in the folder name.
- `benchmark.json` reports decoding time, model/separation time, output writing time, real-time factor (RTF), and speed relative to audio duration.

**Run the same input twice on each system**: the first run may include downloading weights and initializing the ONNX session. The weights are stored in `./models` across runs; output remains in `./output`. Higher speed (lower RTF) is better.

## Caveats

- CPU mode does **not** use AMD integrated graphics. That will require a separate benchmark backend.
- The GPU must successfully initialize the CUDA execution provider; simply detecting an NVIDIA device doesn't prove that every operation ran on the GPU. Monitor `nvidia-smi` and runtime logs.
- The scripts process the whole track in RAM, so very long recordings can exhaust memory or exceed the FFmpeg decode buffer.
- This project has not yet been verified by running a complete Docker build and inference test on the target machines.
- To avoid large Docker build contexts, `.dockerignore` excludes audio, models, and output. Those directories are instead mounted as volumes at runtime.

## Sources

- [@audio/neural-separate](https://www.npmjs.com/package/@audio/neural-separate) — SCNet inference, overlap-add and model download
- [ONNX Runtime Node.js](https://www.npmjs.com/package/onnxruntime-node) — CPU/CUDA providers
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
