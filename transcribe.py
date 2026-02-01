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
    import torch

    # Fix for PyTorch 2.6+ weights_only security change
    # Monkey-patch torch.load to force weights_only=False for pyannote models
    # This is safe for models from trusted sources like Hugging Face
    _original_torch_load = torch.load
    def _patched_torch_load(*args, **kwargs):
        # Force weights_only to False regardless of what was passed
        kwargs['weights_only'] = False
        return _original_torch_load(*args, **kwargs)
    torch.load = _patched_torch_load

    # Also add the safe globals as a backup
    if hasattr(torch.serialization, 'add_safe_globals'):
        torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])

    from pyannote.audio import Pipeline
    import cv2
    import easyocr
    from collections import Counter
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


def transcribe_audio(audio_path: str, model_name: str = "base", language: str = None, device: str = None) -> Dict[str, Any]:
    """
    Transcribe audio file using Whisper model.

    Args:
        audio_path: Path to audio file
        model_name: Whisper model size (tiny, base, small, medium, large)
        language: Optional language code (auto-detect if None)
        device: Device to use ('cuda' or 'cpu', auto-detect if None)

    Returns:
        Dictionary containing transcription results with segments and metadata
    """
    # Auto-detect device if not specified
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading Whisper model '{model_name}' on {device.upper()}...")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Note: GPU not available or not detected. Using CPU (slower).")
    print("(First run will download the model, this may take a few minutes)")

    model = whisper.load_model(model_name, device=device)

    print("Transcribing audio...")
    options = {"verbose": False}
    if language:
        options["language"] = language

    result = model.transcribe(audio_path, **options)
    print("Transcription complete!")

    return result


def diarize_audio(audio_path: str, hf_token: str = None, device: str = None) -> Dict[str, Any]:
    """
    Perform speaker diarization on audio file using pyannote.audio.

    Args:
        audio_path: Path to audio file
        hf_token: Hugging Face access token (required for model access)
        device: Device to use ('cuda' or 'cpu', auto-detect if None)

    Returns:
        Dictionary mapping time ranges to speaker labels
    """
    if not hf_token:
        print("Warning: No Hugging Face token provided. Skipping speaker diarization.")
        print("To enable speaker diarization:")
        print("  1. Visit https://huggingface.co/pyannote/speaker-diarization-3.1")
        print("  2. Accept the model conditions")
        print("  3. Create a token at https://huggingface.co/settings/tokens")
        print("  4. Use --hf-token YOUR_TOKEN")
        return None

    try:
        # Auto-detect device if not specified
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading speaker diarization model on {device.upper()}...")
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token
        )

        # Move pipeline to device
        if device == "cuda":
            pipeline = pipeline.to(torch.device("cuda"))

        print("Identifying speakers...")
        diarization = pipeline(audio_path)

        # Convert diarization to a list of speaker segments
        speaker_segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speaker_segments.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker
            })

        print(f"Speaker diarization complete! Found {len(set(seg['speaker'] for seg in speaker_segments))} unique speakers.")
        return speaker_segments

    except Exception as e:
        print(f"Warning: Speaker diarization failed: {e}")
        print("Continuing without speaker labels...")
        return None


def assign_speakers_to_segments(transcription: Dict[str, Any], speaker_segments: list) -> Dict[str, Any]:
    """
    Assign speaker labels to transcription segments based on timestamp overlap.

    Args:
        transcription: Whisper transcription result with segments
        speaker_segments: List of speaker segments from diarization

    Returns:
        Updated transcription with speaker labels added to each segment
    """
    if not speaker_segments:
        # No diarization available, assign default speaker
        for segment in transcription['segments']:
            segment['speaker'] = "Speaker"
        return transcription

    for segment in transcription['segments']:
        # Find the speaker with the most overlap for this segment
        segment_start = segment['start']
        segment_end = segment['end']
        segment_mid = (segment_start + segment_end) / 2

        # Find which speaker is speaking at the midpoint of this segment
        assigned_speaker = "Unknown"
        for spk_seg in speaker_segments:
            if spk_seg['start'] <= segment_mid <= spk_seg['end']:
                assigned_speaker = spk_seg['speaker']
                break

        segment['speaker'] = assigned_speaker

    return transcription


def extract_zoom_names(video_path: str, device: str = None, verbose: bool = False) -> Dict[str, str]:
    """
    Extract speaker names from Zoom video using OCR.

    Args:
        video_path: Path to the video file
        device: Device to use ('cuda' or 'cpu', auto-detect if None)
        verbose: Print detailed detection information

    Returns:
        Dictionary mapping speaker IDs (SPEAKER_00, SPEAKER_01, etc.) to names
    """
    try:
        # Auto-detect device if not specified
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        use_gpu = (device == "cuda")
        print(f"Extracting names from Zoom video using {device.upper()}...")

        # Initialize OCR reader with GPU support
        reader = easyocr.Reader(['en'], gpu=use_gpu)

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Warning: Could not open video file for name extraction")
            return {}

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Sample frames every 15 seconds (names rarely change in Zoom)
        frame_interval = int(fps * 15)
        detected_names = []
        all_detected_text = []

        frame_count = 0
        frames_processed = 0
        stable_count = 0
        last_names_set = set()

        while frame_count < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
            ret, frame = cap.read()

            if not ret:
                break

            frames_processed += 1

            # Crop to bottom 20% of frame where Zoom name labels appear
            crop_top = int(frame_height * 0.80)
            cropped_frame = frame[crop_top:, :]

            # Perform OCR on cropped region only
            results = reader.readtext(cropped_frame, detail=0)

            if verbose:
                print(f"\n--- Frame {frame_count} ({frame_count/fps:.1f}s) ---")
                print(f"Raw OCR results: {results}")

            # Filter for likely names (avoid UI elements, timestamps, etc.)
            for text in results:
                text = text.strip()
                all_detected_text.append(text)

                # Filter: 2-30 chars, contains letters
                if 2 <= len(text) <= 30 and any(c.isalpha() for c in text):
                    # Skip common Zoom UI elements and short noise
                    skip_terms = ['mute', 'unmute', 'stop video', 'share', 'participants',
                                'chat', 'record', 'reactions', 'leave', 'end', 'more',
                                'view', 'security', 'start', 'stop', 'pause', 'gallery',
                                'speaker', 'zoom', 'meeting', 'host', 'co-host']

                    if text.lower() not in skip_terms and not text.isdigit():
                        detected_names.append(text)
                        if verbose:
                            print(f"  ✓ Accepted: '{text}'")

            frame_count += frame_interval

            # Early stopping: if we've seen the same set of names 3 times in a row, stop
            current_names = set(n for n, c in Counter(detected_names).items() if c >= 2)
            if current_names and current_names == last_names_set:
                stable_count += 1
                if stable_count >= 3:
                    if verbose:
                        print(f"Names stable for 3 consecutive checks, stopping early.")
                    break
            else:
                stable_count = 0
                last_names_set = current_names

        cap.release()

        print(f"Processed {frames_processed} frames")

        # Count occurrences and get names
        if detected_names:
            name_counts = Counter(detected_names)

            if verbose:
                print(f"\n--- Name Counts ---")
                for name, count in name_counts.most_common():
                    print(f"  {name}: {count}")

            # Get unique names that appear at least once (lowered threshold)
            # But prioritize names that appear multiple times
            frequent_names = [name for name, count in name_counts.items() if count >= 1]

            if frequent_names:
                # Sort by frequency (most common first), then alphabetically
                frequent_names_sorted = sorted(frequent_names, key=lambda x: (-name_counts[x], x))
                print(f"Detected {len(frequent_names_sorted)} speaker names from video: {', '.join(frequent_names_sorted)}")
                return {f"SPEAKER_{i:02d}": name for i, name in enumerate(frequent_names_sorted)}
            else:
                print("Warning: Could not reliably detect speaker names from video")
                if verbose:
                    print(f"All detected text: {set(all_detected_text)}")
                return {}
        else:
            print("Warning: No names detected in video frames")
            if verbose:
                print(f"All detected text: {set(all_detected_text)}")
            return {}

    except Exception as e:
        print(f"Warning: Failed to extract names from video: {e}")
        return {}


def map_zoom_names_to_speakers(transcription: Dict[str, Any], zoom_names: Dict[str, str]) -> Dict[str, Any]:
    """
    Replace speaker IDs with actual names extracted from Zoom video.

    Args:
        transcription: Transcription with speaker labels (SPEAKER_00, SPEAKER_01, etc.)
        zoom_names: Dictionary mapping speaker IDs to real names

    Returns:
        Updated transcription with real names
    """
    if not zoom_names:
        return transcription

    for segment in transcription['segments']:
        speaker_id = segment.get('speaker', 'Speaker')
        if speaker_id in zoom_names:
            segment['speaker'] = zoom_names[speaker_id]

    return transcription


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


def format_timestamp_simple(seconds: float) -> str:
    """Convert seconds to simple timestamp format [HH:MM:SS]."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"[{hours:02d}:{minutes:02d}:{secs:02d}]"


def save_txt(result: Dict[str, Any], output_path: str) -> None:
    """Save transcription as plain text with timestamps and optional speaker labels."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for segment in result['segments']:
            timestamp = format_timestamp_simple(segment['start'])
            text = segment['text'].strip()

            # Include speaker label only if present
            if 'speaker' in segment:
                speaker = segment['speaker']
                f.write(f"{timestamp} {speaker}: {text}\n")
            else:
                f.write(f"{timestamp} {text}\n")
    print(f"Transcription saved to: {output_path}")


def save_srt(result: Dict[str, Any], output_path: str) -> None:
    """Save transcription as SRT subtitle file with optional speaker labels."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, segment in enumerate(result['segments'], start=1):
            f.write(f"{i}\n")
            f.write(f"{format_timestamp(segment['start'])} --> {format_timestamp(segment['end'])}\n")

            # Include speaker label only if present
            if 'speaker' in segment:
                f.write(f"{segment['speaker']}: {segment['text'].strip()}\n\n")
            else:
                f.write(f"{segment['text'].strip()}\n\n")
    print(f"SRT subtitles saved to: {output_path}")


def save_vtt(result: Dict[str, Any], output_path: str) -> None:
    """Save transcription as WebVTT subtitle file with optional speaker labels."""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("WEBVTT\n\n")
        for segment in result['segments']:
            f.write(f"{format_timestamp_vtt(segment['start'])} --> {format_timestamp_vtt(segment['end'])}\n")

            # Include speaker label only if present
            if 'speaker' in segment:
                f.write(f"{segment['speaker']}: {segment['text'].strip()}\n\n")
            else:
                f.write(f"{segment['text'].strip()}\n\n")
    print(f"VTT subtitles saved to: {output_path}")


def save_json(result: Dict[str, Any], output_path: str, model_name: str, video_path: str) -> None:
    """Save transcription as JSON with full metadata including optional speakers."""
    segments = []
    for seg in result['segments']:
        segment_data = {
            "start": seg['start'],
            "end": seg['end'],
            "text": seg['text'].strip()
        }
        # Include speaker only if present
        if 'speaker' in seg:
            segment_data['speaker'] = seg['speaker']
        segments.append(segment_data)

    output_data = {
        "source_file": video_path,
        "model": model_name,
        "language": result.get('language', 'unknown'),
        "text": result['text'].strip(),
        "segments": segments
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

    parser.add_argument(
        "--hf-token",
        type=str,
        help="Hugging Face access token for speaker diarization. Get token at https://huggingface.co/settings/tokens"
    )

    parser.add_argument(
        "--no-diarize",
        action="store_true",
        help="Disable speaker diarization (use generic 'Speaker' label)"
    )

    parser.add_argument(
        "--use-zoom-names",
        action="store_true",
        help="Extract speaker names from Zoom video using OCR (requires opencv-python and easyocr)"
    )

    parser.add_argument(
        "--no-speakers",
        action="store_true",
        help="Disable all speaker identification (no speaker labels in output)"
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu", "auto"],
        default="auto",
        help="Device to use for processing: 'cuda' (GPU), 'cpu', or 'auto' (default: auto-detect)"
    )

    parser.add_argument(
        "--verbose-ocr",
        action="store_true",
        help="Show detailed OCR detection output for debugging Zoom name extraction"
    )

    args = parser.parse_args()

    # Print GPU availability info
    print(f"\n{'='*60}")
    print("System Information:")
    print(f"{'='*60}")
    if torch.cuda.is_available():
        print(f"✓ GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  Device Count: {torch.cuda.device_count()}")
    else:
        print("✗ GPU Not Available - Using CPU (slower)")
        print("  To enable GPU:")
        print("  1. Install NVIDIA GPU drivers")
        print("  2. Install CUDA toolkit")
        print("  3. Reinstall PyTorch with CUDA support")
    print(f"{'='*60}\n")

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
        # Determine device to use
        if args.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = args.device

        # Verify CUDA is available if requested
        if device == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA requested but not available. Falling back to CPU.")
            device = "cpu"

        # Extract audio from video
        extract_audio(str(input_path), temp_audio_path)

        # Transcribe audio
        result = transcribe_audio(temp_audio_path, args.model, args.language, device)

        # Perform speaker identification if not disabled
        if not args.no_speakers:
            # Perform speaker diarization if not disabled
            speaker_segments = None
            if not args.no_diarize:
                speaker_segments = diarize_audio(temp_audio_path, args.hf_token, device)

            # Assign speakers to transcription segments
            result = assign_speakers_to_segments(result, speaker_segments)

            # Extract and map Zoom names if requested
            if args.use_zoom_names:
                zoom_names = extract_zoom_names(str(input_path), device, args.verbose_ocr)
                result = map_zoom_names_to_speakers(result, zoom_names)

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
