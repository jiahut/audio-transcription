# audio-transcription

WhisperX + Pyannote: 音频转录（Whisper）+ 说话人分离（Pyannote）+ 词级时间戳对齐（WhisperX）。

## 安装（uv）

```bash
uv venv
uv pip install -r requirements.txt
```

或直接使用 `pyproject.toml`：

```bash
uv pip install -e .
```

## 快速开始

```bash
audio-transcribe ./meeting.m4a \\
  --device auto \\
  --model large-v3 \\
  --vad-method pyannote \\
  --diarize \\
  --hf-token "$HF_TOKEN" \\
  --output-dir ./outputs \\
  --format json --format txt --format srt
```

输出文件默认命名为：`outputs/<音频文件名>.(json|txt|srt|vtt)`

## 配置文件

支持 YAML / JSON，通过 `--config` 指定（CLI 参数会覆盖配置文件）。

示例：`config.yaml`

```yaml
device: auto
device_index: 0
model: large-v3
compute_type: float16
batch_size: 16
language: auto
task: transcribe
trust_checkpoints: true

vad_method: pyannote
vad_options:
  vad_onset: 0.5
  vad_offset: 0.363

align: true
align_model: null

diarize: true
diarize_model: null
num_speakers: null
min_speakers: null
max_speakers: null

formats: [json, txt, srt]
output_dir: ./outputs
```

```bash
audio-transcribe ./meeting.m4a --config config.yaml --hf-token "$HF_TOKEN"
```

## 常用参数

- `--device auto|cuda|cpu`：默认 `auto`
- `--compute-type float16|float32|int8`：CPU 通常用 `int8` 更省内存
- `--diarize` + `--hf-token`：开启说话人分离（需要 HuggingFace Token）
- `--vad-method silero|pyannote`：默认 `pyannote`（`silero` 会通过 `torch.hub` 访问 GitHub，离线/证书环境更容易失败）
- `--ca-bundle /path/to/ca.pem`：HTTPS 证书校验失败时，指定自定义 CA（会设置 `SSL_CERT_FILE/REQUESTS_CA_BUNDLE`）

## 常见报错

### NLTK: `Resource punkt_tab not found` / SSL 证书失败

WhisperX 的 `align`（词级时间戳对齐）依赖 NLTK 的分句模型数据。如果环境无法联网或 HTTPS 证书校验失败，会在对齐阶段报错。

- 方案 1（推荐）：提前下载 NLTK 数据
  - `python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"`
  - 如需指定目录：`NLTK_DATA=/path/to/nltk_data python -c "import nltk; nltk.download('punkt', download_dir='$NLTK_DATA'); nltk.download('punkt_tab', download_dir='$NLTK_DATA')"`
- 方案 2：证书环境问题时，使用 `--ca-bundle /path/to/ca.pem`
- 方案 3：不需要词级对齐时，直接关闭：`--no-align`
