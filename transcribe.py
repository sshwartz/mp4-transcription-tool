#!/usr/bin/env python3
"""
MP4 Transcription Tool using OpenAI Whisper
Transcribes audio from MP4 files to text using local Whisper model.
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, Any

try:
    import whisper
    import ffmpeg
except ImportError as e:
    print(f"Error: Missing required dependency - {e}")
    print("Please install requirements: pip install -r requirements.txt")
    sys.exit(1)


def check_ffmpeg():
    """Check if ffmpeg is installed and available."""
    try:
        ffmpeg.probe("")
    except ffmpeg.Error:
        pass
    except FileNotFoundError:
        print("Error: ffmpeg not found. Please install ffmpeg:")
        print("  Ubuntu/Debian: sudo apt install ffmpeg")
        print("  macOS: brew install ffmpeg")
        print("  Windows: download from https://ffmpeg.org/download.html")
        sys.exit(1)


def extract_audio(video_path: str, output_path: str) -> None:
    """
    Extract audio from MP4 file to WAV format.

    Args:
        video_path: Path to input MP4 file
        output_path: Path to output WAV file
    """
    try:
        print(f"Extracting audio from {video_path}...")
        (
            ffmpeg
            .input(video_path)
            .output(output_path, acodec='pcm_s16le', ac=1, ar='16k')
            .overwrite_output()
            .run(quiet=True)
        )
        print("Audio extraction complete.")
    except ffmpeg.Error as e:
        print(f"Error extracting audio: {e.stderr.decode() if e.stderr else str(e)}")
        sys.exit(1)


def transcribe_audio(audio_path: str, model_name: str = "base", language: str = None) -> Dict[str, Any]:
    """
    Transcribe audio file using Whisper model.

    Args:
        audio_path: Path to audio file
        model_name: Whisper model size (tiny, base, small, medium, large)
        language: Optional language code (auto-detect if None)

    Returns:
        Dictionary containing transcription results with segments and metadata
    """
    print(f"Loading Whisper model '{model_name}'...")
    print("(First run will download the model, this may take a few minutes)")

    model = whisper.load_model(model_name)

    print("Transcribing audio...")
    options = {"verbose": False}
    if language:
        options["language"] = language

    result = model.transcribe(audio_path, **options)
    print("Transcription complete!")

    return result


def format_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def format_timestamp_vtt(seconds: float) -> str:
    """Convert seconds to WebVTT timestamp format (HH:MM:SS.mmm)."""
    timestamp = format_timestamp(seconds)
    return timestamp.replace(',', '.')


def save_txt(result: Dict[str, Any], output_path: str) -> None:
    """Save transcription as plain text."""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(result['text'].strip())
    print(f"Transcription saved to: {output_path}")


def save_srt(result: Dict[str, Any], output_path: str) -> None:
    """Save transcription as SRT subtitle file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, segment in enumerate(result['segments'], start=1):
            f.write(f"{i}\n")
            f.write(f"{format_timestamp(segment['start'])} --> {format_timestamp(segment['end'])}\n")
            f.write(f"{segment['text'].strip()}\n\n")
    print(f"SRT subtitles saved to: {output_path}")


def save_vtt(result: Dict[str, Any], output_path: str) -> None:
    """Save transcription as WebVTT subtitle file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("WEBVTT\n\n")
        for segment in result['segments']:
            f.write(f"{format_timestamp_vtt(segment['start'])} --> {format_timestamp_vtt(segment['end'])}\n")
            f.write(f"{segment['text'].strip()}\n\n")
    print(f"VTT subtitles saved to: {output_path}")


def save_json(result: Dict[str, Any], output_path: str, model_name: str, video_path: str) -> None:
    """Save transcription as JSON with full metadata."""
    output_data = {
        "source_file": video_path,
        "model": model_name,
        "language": result.get('language', 'unknown'),
        "text": result['text'].strip(),
        "segments": [
            {
                "start": seg['start'],
                "end": seg['end'],
                "text": seg['text'].strip()
            }
            for seg in result['segments']
        ]
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"JSON transcription saved to: {output_path}")


def main():
    """Main entry point for the transcription tool."""
    parser = argparse.ArgumentParser(
        description="Transcribe MP4 files using OpenAI Whisper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s video.mp4
  %(prog)s video.mp4 -o transcript.txt
  %(prog)s video.mp4 --model medium --format srt
  %(prog)s video.mp4 --language en --format json
        """
    )

    parser.add_argument(
        "input_file",
        type=str,
        help="Path to MP4 file to transcribe"
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Output file path (default: input_file with appropriate extension)"
    )

    parser.add_argument(
        "-m", "--model",
        type=str,
        choices=["tiny", "base", "small", "medium", "large"],
        default="base",
        help="Whisper model size (default: base). Larger models are more accurate but slower."
    )

    parser.add_argument(
        "-l", "--language",
        type=str,
        help="Source language code (e.g., 'en', 'es', 'fr'). Auto-detect if not specified."
    )

    parser.add_argument(
        "-f", "--format",
        type=str,
        choices=["txt", "srt", "vtt", "json"],
        default="txt",
        help="Output format (default: txt)"
    )

    args = parser.parse_args()

    # Validate input file
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file not found: {args.input_file}")
        sys.exit(1)

    if not input_path.suffix.lower() in ['.mp4', '.mov', '.avi', '.mkv', '.m4a']:
        print(f"Warning: Input file is not a common video format: {input_path.suffix}")

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_path = input_path.with_suffix(f".{args.format}")

    # Check dependencies
    check_ffmpeg()

    # Create temporary file for audio extraction
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
        temp_audio_path = temp_audio.name

    try:
        # Extract audio from video
        extract_audio(str(input_path), temp_audio_path)

        # Transcribe audio
        result = transcribe_audio(temp_audio_path, args.model, args.language)

        # Save output in requested format
        if args.format == "txt":
            save_txt(result, output_path)
        elif args.format == "srt":
            save_srt(result, output_path)
        elif args.format == "vtt":
            save_vtt(result, output_path)
        elif args.format == "json":
            save_json(result, output_path, args.model, str(input_path))

        # Print summary
        print(f"\nSummary:")
        print(f"  Input: {input_path}")
        print(f"  Output: {output_path}")
        print(f"  Model: {args.model}")
        print(f"  Language: {result.get('language', 'auto-detected')}")
        print(f"  Format: {args.format}")

    finally:
        # Clean up temporary audio file
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)


if __name__ == "__main__":
    main()
