# MP4 Transcription Tool

A Python CLI application that transcribes audio from MP4 (and other video/audio) files using OpenAI's Whisper model running locally.

## Features

- **Local Processing**: Uses OpenAI Whisper models running on your machine - no API keys or cloud services required
- **Multiple Output Formats**: Export as plain text, SRT subtitles, WebVTT, or JSON
- **Multiple Model Sizes**: Choose from tiny, base, small, medium, or large models to balance speed vs accuracy
- **Language Support**: Auto-detect language or specify manually (supports 50+ languages)
- **No Usage Limits**: Process as many files as you want, completely free

## Prerequisites

### System Requirements

- Python 3.8 or higher
- ffmpeg (for audio extraction)
- 4GB RAM minimum (8GB+ recommended for larger models)
- GPU with CUDA support optional but recommended for faster processing

### Install ffmpeg

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Windows:**
Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH

## Installation

1. Clone or download this repository
2. Install Python dependencies:

```bash
pip install -r requirements.txt
```

On first run, the Whisper model will be automatically downloaded (~140MB for base model).

## Usage

### Basic Usage

Transcribe an MP4 file to text:
```bash
python transcribe.py video.mp4
```

This creates `video.txt` with the transcription.

### Specify Output File

```bash
python transcribe.py video.mp4 -o transcript.txt
```

### Choose Model Size

```bash
python transcribe.py video.mp4 --model medium
```

Available models (speed vs accuracy tradeoff):
- `tiny` - Fastest, least accurate (~39M parameters)
- `base` - Good balance (default, ~74M parameters)
- `small` - Better accuracy (~244M parameters)
- `medium` - High accuracy (~769M parameters)
- `large` - Best accuracy, slowest (~1550M parameters)

### Specify Language

```bash
python transcribe.py video.mp4 --language en
```

Use ISO 639-1 language codes (en, es, fr, de, etc.). If not specified, language is auto-detected.

### Output Formats

**Plain Text:**
```bash
python transcribe.py video.mp4 --format txt
```

**SRT Subtitles:**
```bash
python transcribe.py video.mp4 --format srt
```

**WebVTT Subtitles:**
```bash
python transcribe.py video.mp4 --format vtt
```

**JSON with Timestamps:**
```bash
python transcribe.py video.mp4 --format json
```

### Complete Example

```bash
python transcribe.py interview.mp4 \
  --output interview_transcript.srt \
  --model medium \
  --language en \
  --format srt
```

## Command Reference

```
usage: transcribe.py [-h] [-o OUTPUT] [-m {tiny,base,small,medium,large}]
                     [-l LANGUAGE] [-f {txt,srt,vtt,json}]
                     input_file

Transcribe MP4 files using OpenAI Whisper

positional arguments:
  input_file            Path to MP4 file to transcribe

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        Output file path (default: input_file with
                        appropriate extension)
  -m {tiny,base,small,medium,large}, --model {tiny,base,small,medium,large}
                        Whisper model size (default: base). Larger models are
                        more accurate but slower.
  -l LANGUAGE, --language LANGUAGE
                        Source language code (e.g., 'en', 'es', 'fr'). Auto-
                        detect if not specified.
  -f {txt,srt,vtt,json}, --format {txt,srt,vtt,json}
                        Output format (default: txt)
```

## Supported Input Formats

While designed for MP4, the tool supports any video/audio format that ffmpeg can read:
- MP4, MOV, AVI, MKV
- M4A, MP3, WAV
- And many more

## Performance Notes

- **GPU Acceleration**: If you have an NVIDIA GPU with CUDA, transcription will be significantly faster
- **Model Size**: The `base` model is recommended for most use cases. Use `medium` or `large` for higher accuracy if you have the hardware
- **Memory**: Larger models require more RAM. If you run out of memory, use a smaller model

## Troubleshooting

**"ffmpeg not found"**
- Install ffmpeg as described in Prerequisites section

**"CUDA out of memory"**
- Use a smaller model with `--model tiny` or `--model base`
- Close other applications using GPU

**Slow transcription**
- Use a smaller model for faster processing
- Enable GPU acceleration if available
- The first run downloads the model and may take longer

**Import errors**
- Ensure you've installed requirements: `pip install -r requirements.txt`
- Check Python version: `python --version` (need 3.8+)

## License

This tool uses OpenAI Whisper, which is released under the MIT License.
