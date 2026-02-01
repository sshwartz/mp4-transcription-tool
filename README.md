# MP4 Transcription Tool

A Python CLI application that transcribes audio from MP4 (and other video/audio) files using OpenAI's Whisper model running locally.

## Features

- **Local Processing**: Uses OpenAI Whisper models running on your machine - no API keys or cloud services required
- **Speaker Diarization**: Automatically identifies and labels different speakers in the audio
- **Multiple Output Formats**: Export as plain text, SRT subtitles, WebVTT, or JSON (all with speaker labels)
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

### Speaker Diarization Setup (Optional)

To enable automatic speaker identification:

1. **Accept Model License**: Visit [pyannote speaker-diarization](https://huggingface.co/pyannote/speaker-diarization-3.1) and accept the model conditions

2. **Create Access Token**:
   - Go to [Hugging Face Settings](https://huggingface.co/settings/tokens)
   - Create a new access token with read permissions
   - Copy the token

3. **Use Token with Tool**:
   ```bash
   python transcribe.py video.mp4 --hf-token YOUR_TOKEN_HERE
   ```

Without a token, the tool will still work but will use a generic "Speaker" label for all segments.

## Usage

### Basic Usage

Transcribe an MP4 file to text:
```bash
python transcribe.py video.mp4
```

This creates `video.txt` with the transcription. Each line will include a timestamp and speaker label.

The tool automatically detects and uses your GPU if available (much faster than CPU).

### With Speaker Diarization

Enable automatic speaker identification:
```bash
python transcribe.py interview.mp4 --hf-token YOUR_TOKEN
```

Output format (TXT):
```
[00:00:05] SPEAKER_00: Welcome to the show.
[00:00:08] SPEAKER_01: Thanks for having me.
[00:00:12] SPEAKER_00: Let's get started with the first question.
```

### Extract Names from Zoom Video

Automatically extract speaker names from Zoom video using OCR:
```bash
python transcribe.py zoom_meeting.mp4 --hf-token YOUR_TOKEN --use-zoom-names
```

Output with real names:
```
[00:00:05] John Smith: Welcome to the meeting.
[00:00:08] Sarah Johnson: Thanks for having me.
[00:00:12] John Smith: Let's review the agenda.
```

**Debug OCR detection:**
If not all names are detected, use verbose mode to see what text is being found:
```bash
python transcribe.py zoom_meeting.mp4 --hf-token YOUR_TOKEN --use-zoom-names --verbose-ocr
```

This works by:
- Scanning the bottom 20% of video frames where Zoom name labels appear
- Sampling frames every 15 seconds (names rarely change in Zoom)
- Stopping early once detected names stabilize across consecutive samples
- Using OCR (EasyOCR) to detect text, filtering out UI elements (Mute, Share, etc.)
- Mapping detected names to speaker IDs from audio diarization
- GPU-accelerated OCR when available for faster processing

**Note**: Works with both Gallery View and Speaker View. Accuracy depends on video quality, name visibility, and font size.

### Without Speaker Identification

Skip all speaker identification (no speaker labels in output):
```bash
python transcribe.py video.mp4 --no-speakers
```

Output without any speaker labels:
```
[00:00:05] Welcome to the meeting.
[00:00:08] Thanks for having me.
[00:00:12] Let's review the agenda.
```

### Skip Audio Diarization Only

Skip audio-based speaker diarization but still use generic labels:
```bash
python transcribe.py video.mp4 --no-diarize
```

Output will use generic "Speaker" label for all segments.

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

### GPU Acceleration

The tool automatically uses your GPU if available (5-10x faster than CPU):

```bash
# Auto-detect and use GPU if available (default)
python transcribe.py video.mp4

# Force GPU usage
python transcribe.py video.mp4 --device cuda

# Force CPU usage (for testing or compatibility)
python transcribe.py video.mp4 --device cpu
```

**Check GPU availability:** The tool displays GPU information at startup:
```
============================================================
System Information:
============================================================
✓ GPU Available: NVIDIA GeForce RTX 3080
  CUDA Version: 11.8
  Device Count: 1
============================================================
```

If you see "GPU Not Available", you may need to:
1. Install NVIDIA GPU drivers
2. Install CUDA toolkit (11.8 or compatible)
3. Reinstall PyTorch with CUDA support: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`

## Command Reference

```
usage: transcribe.py [-h] [-o OUTPUT] [-m {tiny,base,small,medium,large}]
                     [-l LANGUAGE] [-f {txt,srt,vtt,json}]
                     [--hf-token HF_TOKEN] [--no-diarize] [--use-zoom-names]
                     [--no-speakers] [--device {cuda,cpu,auto}] [--verbose-ocr]
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
  --hf-token HF_TOKEN   Hugging Face access token for speaker diarization
  --no-diarize          Disable speaker diarization (use generic 'Speaker' label)
  --use-zoom-names      Extract speaker names from Zoom video using OCR
  --no-speakers         Disable all speaker identification (no speaker labels)
  --device {cuda,cpu,auto}
                        Device to use: 'cuda' (GPU), 'cpu', or 'auto' (default: auto)
  --verbose-ocr         Show detailed OCR detection output for debugging
```

## Supported Input Formats

While designed for MP4, the tool supports any video/audio format that ffmpeg can read:
- MP4, MOV, AVI, MKV
- M4A, MP3, WAV
- And many more

## Performance Notes

- **GPU Acceleration**: Automatically uses NVIDIA GPU if available (5-10x faster than CPU)
  - All components are GPU-accelerated: Whisper transcription, speaker diarization (pyannote), and OCR (EasyOCR)
  - Base model on GPU: ~1000-2000 frames/sec
  - Base model on CPU: ~100-200 frames/sec
  - Use `--device cuda` to force GPU or `--device cpu` to force CPU
- **Model Size**: The `base` model is recommended for most use cases. Use `medium` or `large` for higher accuracy if you have the hardware
- **Memory**: Larger models require more RAM/VRAM. If you run out of memory, use a smaller model
- **CUDA Setup**: Requires CUDA-compatible GPU, NVIDIA drivers, and PyTorch with CUDA support

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

**Speaker diarization not working**
- Ensure you've accepted the model license at https://huggingface.co/pyannote/speaker-diarization-3.1
- Verify your Hugging Face token is valid
- Use `--no-diarize` to skip speaker identification if having issues

## License

This tool uses OpenAI Whisper, which is released under the MIT License.
