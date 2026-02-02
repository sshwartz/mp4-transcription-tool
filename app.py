#!/usr/bin/env python3
"""
Gradio Web Interface for Video Transcription Tool
Wraps transcribe.py functions in a browser-based UI.
"""

import json
import os
import queue
import shutil
import tempfile
import threading
import tkinter as tk
from tkinter import filedialog
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*TorchCodec.*")
warnings.filterwarnings("ignore", message=".*torchaudio.*deprecated.*")

import gradio as gr

from transcribe import (
    check_ffmpeg,
    extract_audio,
    transcribe_audio,
    diarize_audio,
    assign_speakers_to_segments,
    extract_zoom_names,
    map_zoom_names_to_speakers,
    save_txt,
    save_srt,
    save_vtt,
    save_json,
    format_timestamp_simple,
)

SETTINGS_FILE = Path(__file__).parent / "settings.json"

# Temp directory for output files
TEMP_DIR = Path(__file__).parent / "temp"
TEMP_DIR.mkdir(exist_ok=True)
_next_output_num = 1

# Job queue infrastructure
_job_queue = queue.Queue()
_job_list = []  # list of job dicts
_job_list_lock = threading.Lock()
_next_job_id = 1


def _native_save_dialog(title, default_name, filetypes):
    """Open a native OS save dialog on the main thread and return chosen path."""
    result = [None]

    def run():
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.asksaveasfilename(
            title=title,
            initialfile=default_name,
            filetypes=filetypes,
        )
        root.destroy()
        result[0] = path

    # tkinter must run on its own thread to avoid blocking Gradio
    t = threading.Thread(target=run)
    t.start()
    t.join()
    return result[0]


def _native_open_dialog(title, filetypes):
    """Open a native OS open dialog and return chosen path."""
    result = [None]

    def run():
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(
            title=title,
            filetypes=filetypes,
        )
        root.destroy()
        result[0] = path

    t = threading.Thread(target=run)
    t.start()
    t.join()
    return result[0]


def load_settings():
    """Load saved settings from settings.json."""
    defaults = {
        "model": "base",
        "format": "txt",
        "language": "",
        "hf_token": "",
        "use_ocr_names": False,
        "verbose_ocr": False,
    }
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE, "r") as f:
                saved = json.load(f)
            defaults.update(saved)
        except (json.JSONDecodeError, OSError):
            pass
    return defaults


def save_settings(model, fmt, language, hf_token, use_ocr_names, test_mode, verbose_ocr):
    """Save current settings to a user-chosen location via native file dialog."""
    settings = {
        "model": model,
        "format": fmt,
        "language": language,
        "hf_token": hf_token,
        "use_ocr_names": use_ocr_names,
        "test_mode": test_mode,
        "verbose_ocr": verbose_ocr,
    }
    path = _native_save_dialog(
        title="Save Settings",
        default_name="settings.json",
        filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
    )
    if not path:
        return "Save cancelled."
    with open(path, "w") as f:
        json.dump(settings, f, indent=2)
    # Also save to default location so settings load on next startup
    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings, f, indent=2)
    return f"Settings saved to {path}"



def open_settings():
    """Load settings from a user-chosen file via native file dialog."""
    path = _native_open_dialog(
        title="Open Settings",
        filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
    )
    if not path:
        return (gr.update(),) * 7 + ("Open cancelled.",)
    try:
        with open(path, "r") as f:
            settings = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        return (gr.update(),) * 7 + (f"Error loading settings: {e}",)
    # Also save as default settings
    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings, f, indent=2)
    return (
        settings.get("model", "base"),
        settings.get("format", "txt"),
        settings.get("language", ""),
        settings.get("hf_token", ""),
        settings.get("use_ocr_names", False),
        settings.get("test_mode", False),
        settings.get("verbose_ocr", False),
        f"Settings loaded from {path}",
    )


def _run_job(job):
    """Execute the transcription pipeline for a single job."""
    global _next_output_num

    video_path = job["video_path"]
    model = job["model"]
    fmt = job["fmt"]
    language = job["language"]
    hf_token = job["hf_token"]
    use_ocr_names = job["use_ocr_names"]
    test_mode = job["test_mode"]
    verbose_ocr = job["verbose_ocr"]
    cancel_event = job["cancel_event"]

    def check_cancelled():
        if cancel_event.is_set():
            raise RuntimeError("Transcription cancelled by user.")

    check_ffmpeg()

    temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_audio_path = temp_audio.name
    temp_audio.close()

    temp_output = tempfile.NamedTemporaryFile(suffix=f".{fmt}", delete=False)
    temp_output_path = temp_output.name
    temp_output.close()

    try:
        duration = 180 if test_mode else None
        extract_audio(video_path, temp_audio_path, duration=duration)
        check_cancelled()

        lang = language.strip() if language and language.strip() else None
        result = transcribe_audio(temp_audio_path, model, lang)
        check_cancelled()

        if use_ocr_names:
            token = hf_token.strip() if hf_token and hf_token.strip() else None
            speaker_segments = diarize_audio(temp_audio_path, token)
            result = assign_speakers_to_segments(result, speaker_segments)
            check_cancelled()

            zoom_names = extract_zoom_names(video_path, verbose=verbose_ocr)
            result = map_zoom_names_to_speakers(result, zoom_names)
            check_cancelled()

        if fmt == "txt":
            save_txt(result, temp_output_path)
        elif fmt == "srt":
            save_srt(result, temp_output_path)
        elif fmt == "vtt":
            save_vtt(result, temp_output_path)
        elif fmt == "json":
            save_json(result, temp_output_path, model, video_path)

        # Build display text
        lines = []
        for seg in result["segments"]:
            ts = format_timestamp_simple(seg["start"])
            text = seg["text"].strip()
            speaker = seg.get("speaker")
            if speaker:
                lines.append(f"{ts} {speaker}: {text}")
            else:
                lines.append(f"{ts} {text}")
        transcript_text = "\n".join(lines)

        # Save output to temp directory with sequential naming
        output_name = f"temp_output{_next_output_num}.{fmt}"
        _next_output_num += 1
        download_path = str(TEMP_DIR / output_name)
        shutil.copy2(temp_output_path, download_path)
        job["output_path"] = download_path

        job["result"] = transcript_text
        job["status"] = "Done"

    except RuntimeError as e:
        if "cancelled" in str(e).lower():
            job["status"] = "Cancelled"
            job["result"] = "Transcription cancelled."
        else:
            job["status"] = "Error"
            job["result"] = f"Error: {e}"
    except Exception as e:
        job["status"] = "Error"
        job["result"] = f"Error: {e}"
    finally:
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)


def _worker():
    """Background worker that processes jobs from the queue serially."""
    while True:
        job = _job_queue.get()
        if job["cancel_event"].is_set():
            job["status"] = "Cancelled"
            job["result"] = "Transcription cancelled."
            _job_queue.task_done()
            continue
        job["status"] = "Processing"
        _run_job(job)
        _job_queue.task_done()


# Start background worker thread
_worker_thread = threading.Thread(target=_worker, daemon=True)
_worker_thread.start()


def queue_video(video_file, model, fmt, language, hf_token, use_ocr_names, test_mode, verbose_ocr):
    """Add a transcription job to the queue."""
    global _next_job_id
    if video_file is None:
        return "Please upload a video file."

    with _job_list_lock:
        job_id = _next_job_id
        _next_job_id += 1

    filename = Path(video_file).name

    job = {
        "id": job_id,
        "filename": filename,
        "status": "Queued",
        "result": None,
        "cancel_event": threading.Event(),
        "video_path": video_file,
        "model": model,
        "fmt": fmt,
        "language": language,
        "hf_token": hf_token,
        "use_ocr_names": use_ocr_names,
        "test_mode": test_mode,
        "verbose_ocr": verbose_ocr,
    }

    with _job_list_lock:
        _job_list.append(job)

    _job_queue.put(job)
    return f"Job #{job_id} queued: {filename}"


def cancel_job(job_id_str):
    """Cancel a specific job by ID."""
    try:
        job_id = int(job_id_str)
    except (ValueError, TypeError):
        return
    with _job_list_lock:
        for job in _job_list:
            if job["id"] == job_id:
                if job["status"] not in ("Done", "Cancelled", "Error"):
                    job["cancel_event"].set()
                    if job["status"] == "Queued":
                        job["status"] = "Cancelled"
                        job["result"] = "Transcription cancelled."


def delete_job(job_id_str):
    """Remove a job from the job list."""
    try:
        job_id = int(job_id_str)
    except (ValueError, TypeError):
        return
    with _job_list_lock:
        for i, job in enumerate(_job_list):
            if job["id"] == job_id:
                if job["status"] in ("Queued", "Processing"):
                    job["cancel_event"].set()
                _job_list.pop(i)
                return


def get_queue_snapshot():
    """Return a serializable snapshot of the job list for gr.State."""
    with _job_list_lock:
        return [
            {
                "id": job["id"],
                "filename": job["filename"],
                "status": job["status"],
                "output_path": job.get("output_path"),
            }
            for job in _job_list
        ]


def get_latest_result():
    """Return the transcript of the most recently completed job."""
    with _job_list_lock:
        for job in reversed(_job_list):
            if job["status"] == "Done" and job["result"]:
                return job["result"]
    return ""


def refresh_status():
    """Called by timer to update queue snapshot and transcript."""
    return get_queue_snapshot(), get_latest_result()


def browse_video():
    """Open a native file dialog to select a video file."""
    path = _native_open_dialog(
        title="Select Video File",
        filetypes=[
            ("Video files", "*.mp4 *.mkv *.avi *.mov *.webm *.flv *.wmv"),
            ("All files", "*.*"),
        ],
    )
    if not path:
        return gr.update(), "No file selected"
    return path, Path(path).name



def build_app():
    settings = load_settings()

    with gr.Blocks(title="Video Transcription Tool") as app:
        gr.Markdown("# Video Transcription Tool")

        with gr.Row():
            with gr.Column(scale=1):
                video_input = gr.State(value=None)
                upload_btn = gr.Button("Upload Local Video File")
                video_label = gr.Textbox(label="Selected Video", interactive=False, value="No file selected")
                model_dropdown = gr.Dropdown(
                    choices=["tiny", "base", "small", "medium", "large"],
                    value=settings["model"],
                    label="Model Size",
                )
                gr.Markdown("*The larger the model size, the better the transcription quality but larger models require much more time to run. The base model size is usually sufficient.*")
                format_dropdown = gr.Dropdown(
                    choices=["txt", "srt", "vtt", "json"],
                    value=settings["format"],
                    label="Output Format",
                )
                language_input = gr.Textbox(
                    value=settings["language"],
                    label="Language (blank = auto-detect)",
                    placeholder="en, es, fr, ...",
                )
                hf_token_input = gr.Textbox(
                    value=settings["hf_token"],
                    label="HuggingFace Token (for diarization)",
                    type="password",
                )
                use_ocr_cb = gr.Checkbox(value=settings.get("use_ocr_names", False), label="Use OCR to Add Speaker Names")
                test_mode_cb = gr.Checkbox(value=settings.get("test_mode", False), label="Test Using First 3 Minutes Only")
                verbose_ocr_cb = gr.Checkbox(value=settings["verbose_ocr"], label="Verbose OCR (for debugging only)")

                with gr.Row():
                    transcribe_btn = gr.Button("Transcribe", variant="primary")
                    open_settings_btn = gr.Button("Open Settings")
                    save_settings_btn = gr.Button("Save Settings")

            with gr.Column(scale=2):
                transcript_output = gr.Textbox(label="Transcript", lines=25)

                gr.Markdown("### Queue Status")
                queue_state = gr.State(value=[])

                @gr.render(inputs=[queue_state])
                def render_queue(jobs):
                    if not jobs:
                        gr.Markdown("No jobs yet.")
                        return
                    for job in jobs:
                        with gr.Row():
                            gr.Textbox(
                                value=f"#{job['id']}  {job['filename']}  —  {job['status']}",
                                interactive=False,
                                show_label=False,
                                scale=4,
                            )
                            if job["status"] in ("Queued", "Processing"):
                                cancel_btn = gr.Button("Cancel", variant="stop", scale=1, key=f"cancel-{job['id']}")
                                cancel_btn.click(
                                    fn=lambda jid=str(job["id"]): cancel_job(jid),
                                    inputs=None,
                                    outputs=[],
                                )
                            if job["status"] == "Done" and job.get("output_path"):
                                dl_btn = gr.DownloadButton(
                                    "Download Transcript",
                                    value=job["output_path"],
                                    scale=1,
                                    key=f"dl-{job['id']}",
                                )
                            del_btn = gr.Button("Delete", scale=1, key=f"del-{job['id']}")
                            del_btn.click(
                                fn=lambda jid=str(job["id"]): delete_job(jid),
                                inputs=None,
                                outputs=[],
                            )

        timer = gr.Timer(value=2)
        timer.tick(
            fn=refresh_status,
            inputs=[],
            outputs=[queue_state, transcript_output],
        )

        upload_btn.click(
            fn=browse_video,
            inputs=[],
            outputs=[video_input, video_label],
        )

        transcribe_btn.click(
            fn=queue_video,
            inputs=[video_input, model_dropdown, format_dropdown, language_input, hf_token_input, use_ocr_cb, test_mode_cb, verbose_ocr_cb],
            outputs=[transcript_output],
        )

        open_settings_btn.click(
            fn=open_settings,
            inputs=[],
            outputs=[model_dropdown, format_dropdown, language_input, hf_token_input, use_ocr_cb, test_mode_cb, verbose_ocr_cb, transcript_output],
        )

        save_settings_btn.click(
            fn=save_settings,
            inputs=[model_dropdown, format_dropdown, language_input, hf_token_input, use_ocr_cb, test_mode_cb, verbose_ocr_cb],
            outputs=[transcript_output],
        )

    return app


if __name__ == "__main__":
    import subprocess

    app = build_app()

    # Start named cloudflared tunnel in background
    proc = subprocess.Popen(
        [
            str(Path(__file__).parent / "cloudflared.exe"),
            "tunnel", "run",
            "--url", "http://127.0.0.1:7860",
            "transcribe",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    # Drain tunnel output in background
    def _drain():
        for line in proc.stdout:
            print(f"[tunnel] {line.strip()}")

    threading.Thread(target=_drain, daemon=True).start()

    print("\n*** Public URL: https://transcribe.aiperspectives.com ***\n")
    app.launch(server_name="127.0.0.1", server_port=7860)
