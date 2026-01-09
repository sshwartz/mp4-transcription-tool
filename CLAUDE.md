# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MP4 Transcription Tool - A Python CLI application that transcribes audio from video files using OpenAI's Whisper model running locally. The tool extracts audio from MP4 files using ffmpeg, processes it through Whisper for speech-to-text conversion, and outputs transcriptions in multiple formats (TXT, SRT, VTT, JSON).

## Development Commands

### Setup
```bash
# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Install ffmpeg (system dependency - required)
# Ubuntu/Debian: sudo apt install ffmpeg
# macOS: brew install ffmpeg
# Windows: download from ffmpeg.org
```

### Running the Tool
```bash
# Activate virtual environment first
source venv/bin/activate

# Basic usage
python transcribe.py video.mp4

# With options
python transcribe.py video.mp4 --model medium --format srt -o output.srt

# Help
python transcribe.py --help
```

### Testing
```bash
# No automated tests currently - manual testing required
# Test with a sample video file:
python transcribe.py test_video.mp4 --format json

# Test different model sizes for accuracy comparison:
python transcribe.py test.mp4 --model tiny -o test_tiny.txt
python transcribe.py test.mp4 --model base -o test_base.txt
python transcribe.py test.mp4 --model medium -o test_medium.txt
```

## Architecture

### Single-File Modular Design
The entire application is contained in `transcribe.py` with a function-based modular architecture. This design choice keeps the codebase simple and maintainable for a CLI tool.

**Processing Pipeline:**
1. **Input Validation** (`main`) → Verify file exists, check dependencies
2. **Audio Extraction** (`extract_audio`) → ffmpeg extracts audio to temporary WAV file (16kHz mono PCM)
3. **Transcription** (`transcribe_audio`) → Whisper model processes audio
4. **Output Formatting** (`save_txt/srt/vtt/json`) → Save results in requested format
5. **Cleanup** → Remove temporary files (always executed in finally block)

**Key Functions:**

- **`check_ffmpeg()`** - Validates ffmpeg installation with helpful error messages per OS
- **`extract_audio(video_path, output_path)`** - Uses ffmpeg-python to convert video to 16kHz mono WAV (Whisper's optimal format)
- **`transcribe_audio(audio_path, model_name, language)`** - Loads Whisper model and transcribes, handles auto language detection
- **`format_timestamp(seconds)` / `format_timestamp_vtt(seconds)`** - Convert float seconds to subtitle timestamp formats (SRT vs VTT have different formats)
- **`save_txt/srt/vtt/json(result, output_path)`** - Format-specific output handlers
- **`main()`** - CLI orchestration using argparse, manages temporary files

### Dependencies

**Python Packages:**
- **openai-whisper** - Core transcription engine (speech-to-text model)
- **ffmpeg-python** - Python bindings for ffmpeg (audio extraction)
- **torch/torchaudio** - PyTorch backend required by Whisper (includes CUDA support for GPU acceleration)

**System Dependencies:**
- **ffmpeg** - Must be installed separately (not in requirements.txt), checked at runtime with helpful per-OS install instructions

### Whisper Model Management

Whisper models are automatically downloaded on first use to `~/.cache/whisper/`:
- **tiny**: ~39M params, ~140MB download - fastest, lowest accuracy
- **base**: ~74M params, ~140MB download - default, good balance
- **small**: ~244M params, ~460MB download - better accuracy
- **medium**: ~769M params, ~1.5GB download - high accuracy, slower
- **large**: ~1550M params, ~2.9GB download - best accuracy, requires GPU for reasonable speed

Models are cached globally per user, not per project. Once downloaded, subsequent runs are much faster.

### Audio Processing Details

**FFmpeg Extraction Settings:**
- **Codec**: `pcm_s16le` (16-bit PCM) - uncompressed audio for maximum quality
- **Channels**: `ac=1` (mono) - Whisper works better with mono audio
- **Sample Rate**: `ar='16k'` (16kHz) - Whisper's expected sample rate
- Temporary WAV files are used (not MP3) to avoid lossy compression artifacts

## Code Patterns

### Error Handling Philosophy
- **Fail fast with helpful messages**: Check system dependencies (ffmpeg) before processing to give immediate actionable feedback
- **OS-specific help**: Provide platform-specific installation instructions in error messages
- **Validate inputs early**: Check file existence before starting extraction to avoid wasting time
- **Always cleanup**: Use try/finally blocks to ensure temporary files are removed even on errors

### Temporary File Management
```python
with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
    temp_audio_path = temp_audio.name

try:
    # Process with temp_audio_path
finally:
    if os.path.exists(temp_audio_path):
        os.remove(temp_audio_path)
```
- Use `delete=False` to control cleanup timing (cleanup in finally block instead of on file close)
- Always use `.wav` suffix for Whisper compatibility
- Remove in finally block to ensure cleanup even on errors

### Output Format Strategy
Each format has a dedicated save function with format-specific logic:
- **TXT**: Plain text only (`result['text']`)
- **SRT**: Subtitles with 1-indexed sequence numbers, HH:MM:SS,mmm timestamps
- **VTT**: WebVTT with header ("WEBVTT\n\n"), HH:MM:SS.mmm timestamps (note: period not comma)
- **JSON**: Full metadata including segments array with timing, detected language, model used

### CLI Design Principles
- **Sensible defaults**: base model, txt format, auto-detect language, same filename with new extension
- **Short and long options**: `-o`/`--output`, `-m`/`--model`, `-f`/`--format`, `-l`/`--language`
- **Examples in help text**: argparse epilog shows common usage patterns
- **Choices validation**: Use `choices=[]` for model sizes and formats to auto-validate user input
- **Path handling**: Use `pathlib.Path` for cross-platform path manipulation

## Performance Considerations

### CPU vs GPU
- **CPU Mode**: Automatic fallback, uses FP32 (warning displayed), slower but works everywhere
- **GPU Mode**: Automatic if CUDA available, uses FP16, significantly faster (5-10x for large models)
- Model loading is one-time per run, actual transcription dominates runtime

### Processing Speed
- **Tiny/Base models**: Can run on CPU reasonably (~1,200 frames/sec on modern CPU)
- **Medium/Large models**: Strongly benefit from GPU acceleration
- First run per model downloads the model (one-time delay)
- Audio extraction is fast (usually <10% of total time)

### Memory Requirements
- Depends on model size and audio length
- Base model: ~4GB RAM sufficient
- Large model: ~8GB+ RAM recommended
- Long videos (>1 hour): May require more memory for segment management

## Known Limitations and Future Improvements

### Current Limitations
- **No batch processing**: Single file per invocation (could add glob pattern support)
- **No progress bars**: Whisper handles progress internally (could expose with verbose flag)
- **Limited video format validation**: Only checks file extension, not actual codec
- **No resume capability**: Failed transcriptions must restart from beginning
- **No speaker diarization**: Cannot identify different speakers in the audio
- **CPU-only is slow**: Large files on CPU can take 30+ minutes

### Potential Enhancements
- Add `--batch` flag to process multiple files with glob patterns
- Add `--verbose` flag to show Whisper's internal progress
- Validate video codec using ffmpeg.probe() before extraction
- Add `--resume` flag with checkpoint saving for long transcriptions
- Add `--diarize` flag using pyannote.audio for speaker identification
- Add quality presets (fast/balanced/accurate) that auto-select model and settings
