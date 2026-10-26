# Third-party components in the V3 release archives

The project's own code is under the Apache License 2.0 (`LICENSE`, `NOTICE`). The release
archives also redistribute the following, each under its own license.

| component | where in the release | license | license text |
|---|---|---|---|
| **Ollama** v0.34.4, linux-arm64 + jetpack5 builds, unmodified, from <https://github.com/ollama/ollama/releases/tag/v0.34.4> | `base/llm/ollama/` | MIT | `base/llm/ollama/LICENSE` |
| llama.cpp and its vendored code (inside Ollama) | `base/llm/ollama/lib/ollama/` | MIT and others | `base/llm/ollama/lib/ollama/LLAMA_CPP_LICENSE`, `LLAMA_CPP_VENDORS_LICENSE` |
| cpp-httplib, Go runtime (inside Ollama) | `base/llm/ollama/` | MIT; BSD-3-Clause | `base/llm/ollama/lib/ollama/CPP_HTTPLIB_LICENSE`, `GO_LICENSE` |
| NVIDIA CUDA runtime and cuBLAS 11 (shipped by Ollama's jetpack5 build) | `base/llm/ollama/lib/ollama/cuda_jetpack5/` | NVIDIA CUDA Toolkit EULA, as redistributable runtime components | <https://docs.nvidia.com/cuda/eula/> |
| **Qwen3-1.7B** (Q4_K_M, as published by Ollama as `qwen3:1.7b`), by the Qwen team, Alibaba Cloud | `base/llm/models/` (second archive) | Apache-2.0 | the model's `license` blob in `base/llm/models/blobs/` (`sha256-d18a5cc7…`) |
| **YOLOX-Nano** (`yolox_nano.onnx`, release 0.1.1rc0) and YOLOX's sample image `dog.jpg`, by Megvii, unmodified, from <https://github.com/Megvii-BaseDetection/YOLOX> | `base/perception/models/` | Apache-2.0 | <https://github.com/Megvii-BaseDetection/YOLOX/blob/main/LICENSE> |

The dance policy (`v3_package/tree/v3_models/dance_wave/`) was trained with BeyondMimic's
motion-tracking code (`whole_body_tracking`, MIT). That code is used for training only and
is not in the release; it is vendored in the source repository under
`v3/third_party/whole_body_tracking/`.

Only the Ollama files needed on a Jetson are included: the `cuda_v12` and `cuda_v13`
directories of the release, which are for data-center GPUs, are left out. Nothing else
was changed.

The Unitree SDK (unitree_sdk2), ROS 2 foxy, JetPack, CUDA and TensorRT are **not**
included. They must already be installed on the robot.
