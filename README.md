<h1 align="center">
  <br>
  silero-vad-go
  <br>
</h1>
<h4 align="center">A simple Golang (CGO + ONNX Runtime) speech detector powered by Silero VAD</h4>
<p align="center">
  <a href="https://pkg.go.dev/github.com/streamer45/silero-vad-go"><img src="https://pkg.go.dev/badge/github.com/streamer45/silero-vad-go.svg" alt="Go Reference"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
</p>
<br>

### Requirements

- [Golang](https://go.dev/doc/install) >= v1.21
- A C compiler (e.g. GCC)
- ONNX Runtime (v1.18.1)
- A [Silero VAD](https://github.com/snakers4/silero-vad) model (v5)

### Development

In order to build and/or run this library, you need to export (or pass) some env variables to point to the ONNX runtime files.

#### Linux

```sh
LD_RUN_PATH="/usr/local/lib/onnxruntime-linux-x64-1.18.1/lib"
LIBRARY_PATH="/usr/local/lib/onnxruntime-linux-x64-1.18.1/lib"
C_INCLUDE_PATH="/usr/local/include/onnxruntime-linux-x64-1.18.1/include"
```

#### Darwin (MacOS)

```sh
LIBRARY_PATH="/usr/local/lib/onnxruntime-linux-x64-1.18.1/lib"
C_INCLUDE_PATH="/usr/local/include/onnxruntime-linux-x64-1.18.1/include"
sudo update_dyld_shared_cache
```

### License

MIT License - see [LICENSE](LICENSE) for full text

