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
export HF_TOKEN="your_huggingface_token"

audio-transcribe ./meeting.m4a \\
  --device auto \\
  --model large-v3 \\
  --vad-method pyannote \\
  --diarize \\
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
export HF_TOKEN="your_huggingface_token"
audio-transcribe ./meeting.m4a --config config.yaml
```

## 常用参数

CLI 支持通过 `--config` 读取 YAML/JSON（命令行参数会覆盖配置文件）。下表给出所有参数的默认值与选型建议（默认值以未通过 CLI/配置覆盖时的内置 `AppConfig` 为准）。

| 参数 | 类型/可选值 | 默认值 | 何时用（选择逻辑/背后考虑） |
|---|---|---:|---|
| `audio`（位置参数） | 1+ 个音频文件路径 | （必填） | 输入音频；支持一次传多个文件跑批。 |
| `-h/--help` | 开关 | `false` | 查看帮助。 |
| `--config` | 路径 | `null` | 固化一套可复用配置（团队/跑批/上线），仅用 CLI 覆盖少数字段。 |
| `--dump-config` | 路径 | `null` | 输出“配置文件 + CLI 覆盖”后的最终配置，便于复现与审计。 |
| `--dry-run` | 开关 | `false` | 只校验输入与配置是否有效，不实际跑模型；用于脚本预检查。 |
| `--device` | `auto/cuda/cpu` | `auto` | `auto`=有 GPU 就用；强制 `cpu` 用于无 GPU/显存极紧；强制 `cuda` 用于你确认 GPU 可用且不想被自动逻辑影响。 |
| `--device-index` | int | `0` | 多卡机器指定用哪张卡，避免抢占默认 0 卡。 |
| `--model` | string（如 `large-v3`） | `large-v3` | 质量优先用更大模型（更慢/更吃显存）；速度/成本优先用更小模型。 |
| `--compute-type` | `float16/float32/int8` | `float16` | GPU 常用 `float16`（性能/显存更优）；CPU 常用 `int8`（更省内存/可能更快但略降效果）；追求稳定可用 `float32`（更慢更占）。 |
| `--batch-size` | int | `16` | 显存紧就降以降低 OOM 风险；显存富余可升以提升吞吐。 |
| `--language` | 语言码或 `auto` | `auto` | 明确语言（如 `zh`）通常更稳（减少误判语言）；不确定/混合语用 `auto`。 |
| `--task` | `transcribe/translate` | `transcribe` | `translate` 用于“非英语音频→英文输出”；一般中文会议纪要用 `transcribe`。 |
| `--threads` | int | `4` | 主要影响 CPU 侧；CPU 跑或机器核多可调大；GPU 跑通常不是瓶颈。 |
| `--download-root` | 路径 | `null` | 指定模型/NLTK 等缓存目录（大盘/共享盘/容器挂载卷），便于离线复用与集中管理。 |
| `--local-files-only` | 开关 | `false` | 离线/内网：强制只用本地缓存，避免运行时下载（缓存不全会失败）。 |
| `--ca-bundle` | PEM 路径 | `null` | 公司代理/自签 CA 导致 HTTPS 校验失败时使用；会设置 `SSL_CERT_FILE/REQUESTS_CA_BUNDLE/CURL_CA_BUNDLE`。 |
| `--vad-method` | `pyannote/silero` | `pyannote` | `pyannote` 更适合 GitHub 访问不通/证书复杂环境；`silero` 通过 `torch.hub` 走 GitHub，离线/证书环境更易失败。 |
| `--vad-option KEY=VALUE` | 可重复 KV | `{}` | 微调切段敏感度（噪声大/断句异常/切得太碎或漏语音时）；建议小步调整并用同一段音频对比效果。 |
| `--align / --no-align` | 布尔 | `true` | 需要词级时间戳/更精细字幕就开；想更快或 NLTK 资源/联网有问题就关（`--no-align`）。 |
| `--align-model` | string | `null` | 默认自动选择对齐模型；只有要锁定模型版本或特殊语言兼容时才指定。 |
| `--interpolate-method` | string（如 `nearest`） | `nearest` | 对齐时间戳插值策略；一般不改，只有你明确知道某策略更适合你的音频时调整。 |
| `--diarize / --no-diarize` | 布尔 | `false` | 需要区分不同说话人（输出 `SPEAKER_00/01...`）就开；不需要则关以节省时间/资源。 |
| `--diarize-model` | string | `null` | 默认用 whisperx/pyannote 的默认 diarization 模型；只有要固定某模型或做兼容/效果对比时才指定。 |
| `--num-speakers` | int | `null` | 已知就是 N 个人时建议设置：减少过分拆分/合并，提高稳定性。 |
| `--min-speakers` | int | `null` | 不确定人数但有下限时，用于约束聚类范围，避免全合成 1 人。 |
| `--max-speakers` | int | `null` | 不确定人数但有上限时，用于避免“越分越多”。 |
| `--hf-token` | string | `null` | 仅在 `--diarize` 且模型需要鉴权时才需要。安全上更推荐用环境变量 `HF_TOKEN`/`HUGGINGFACE_TOKEN`，避免出现在 shell history/进程列表里。 |
| `--asr-option KEY=VALUE` | 可重复 KV | `{}` | 传递额外 ASR 参数（具体 key 取决于 whisperx/后端支持）；质量优先通常更“强搜索”，速度优先更“保守设置”。 |
| `--trust-checkpoints / --no-trust-checkpoints` | 布尔 | `true` | `true` 更偏兼容优先（应对 PyTorch 2.6+ `weights_only` 行为变化）；强调加载安全且确认依赖兼容时可尝试关闭。 |
| `--output-dir` | 路径 | `./outputs` | 跑批/多任务建议显式指定（按日期/项目分目录），便于归档与避免覆盖。 |
| `--format` | 可重复：`json/txt/srt/vtt` | `json,txt` | `json` 便于二次处理；`txt` 便于阅读；`srt/vtt` 用于字幕；需要多种输出就重复传多个 `--format`。 |
| `--print / --no-print` | 布尔 | `true` | 交互调试开；跑批/日志很大时关（`--no-print`）。 |

## 常见报错

### NLTK: `Resource punkt_tab not found` / SSL 证书失败

WhisperX 的 `align`（词级时间戳对齐）依赖 NLTK 的分句模型数据。如果环境无法联网或 HTTPS 证书校验失败，会在对齐阶段报错。

- 方案 1（推荐）：提前下载 NLTK 数据
  - `python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"`
  - 如需指定目录：`NLTK_DATA=/path/to/nltk_data python -c "import nltk; nltk.download('punkt', download_dir='$NLTK_DATA'); nltk.download('punkt_tab', download_dir='$NLTK_DATA')"`
- 方案 2：证书环境问题时，使用 `--ca-bundle /path/to/ca.pem`
- 方案 3：不需要词级对齐时，直接关闭：`--no-align`
