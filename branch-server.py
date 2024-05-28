import os
import shutil
import subprocess
import json
import time
import re
import gradio as gr
import yt_dlp
import threading
from datetime import datetime
import queue

# Define global variables and paths
TEMP_DIR = "temp"
OUTPUT_DIR = "output"
PROCESSED_URLS_FILE = "processed_urls.json"
LOG_FILE = "transcription.log"
WHITELIST_FILE = "whitelist.json"
USER_ACTIVITY_FILE = "user_activity.json"
status_queue = queue.Queue()

# Load processed URLs
if os.path.exists(PROCESSED_URLS_FILE):
    with open(PROCESSED_URLS_FILE, "r") as f:
        processed_urls = json.load(f)
else:
    processed_urls = {}

# Load whitelist
if os.path.exists(WHITELIST_FILE):
    with open(WHITELIST_FILE, "r") as f:
        whitelist = json.load(f)
else:
    whitelist = {}

# Load user activity
if os.path.exists(USER_ACTIVITY_FILE):
    with open(USER_ACTIVITY_FILE, "r") as f:
        user_activity = json.load(f)
else:
    user_activity = {}

# Save processed URLs
def save_processed_urls():
    with open(PROCESSED_URLS_FILE, "w") as f:
        json.dump(processed_urls, f)

# Save user activity
def save_user_activity():
    with open(USER_ACTIVITY_FILE, "w") as f:
        json.dump(user_activity, f)

# Check if ffmpeg is installed
def check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        raise RuntimeError("ffmpeg is not installed or not found in PATH")

# Sanitize filename by removing special characters
def sanitize_filename(filename):
    return re.sub(r'[^\w\s-]', '', filename)

# Function to download video using yt-dlp
def download_video(url, progress_callback=None):
    ydl_opts = {
        'outtmpl': os.path.join(TEMP_DIR, '%(title)s.%(ext)s'),
        'format': 'bestvideo[height<=144]+bestaudio/best',  # lowest video quality and best audio quality
        'progress_hooks': [progress_callback] if progress_callback else []
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        sanitized_title = sanitize_filename(info['title'])
        new_file_path = os.path.join(TEMP_DIR, f"{sanitized_title}.{info['ext']}")
        os.rename(ydl.prepare_filename(info), new_file_path)
        return new_file_path, info['duration']

# Function to convert video to audio
def convert_video_to_audio(video_path, audio_format='wav'):
    audio_path = os.path.splitext(video_path)[0] + f'.{audio_format}'
    if not os.path.exists(audio_path):
        try:
            subprocess.run(
                ["ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2", audio_path],
                check=True
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"ffmpeg failed to convert video to audio: {e}")
    return audio_path

# Function to process the video and generate transcription
def process_video(file_path, force_reprocess=False, progress_callback=None):
    file_name = os.path.basename(file_path)
    file_base = os.path.splitext(file_name)[0]
    output_json = os.path.join(OUTPUT_DIR, f"{file_base}.json")
    output_srt = os.path.join(OUTPUT_DIR, f"{file_base}.srt")

    if force_reprocess:
        # Delete existing JSON and SRT files if they exist
        if os.path.exists(output_json):
            os.remove(output_json)
        if os.path.exists(output_srt):
            os.remove(output_srt)

    # Check if JSON and SRT files already exist
    if os.path.exists(output_json) and os.path.exists(output_srt):
        return output_json, output_srt

    # Convert video to audio
    audio_path = convert_video_to_audio(file_path)

    # Run the transcription command
    try:
        subprocess.run(
            f'insanely-fast-whisper --file-name "{audio_path}" --model-name openai/whisper-large-v3 --task transcribe --language en --device-id 0 --transcript-path "{output_json}"',
            shell=True,
            check=True
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Transcription failed: {e}")

    # Convert JSON to SRT
    convert_to_srt(output_json, output_srt)

    # Delete original video file to save space
    os.remove(file_path)

    return output_json, output_srt

# Function to convert JSON to SRT
def convert_to_srt(input_path, output_path):
    def format_seconds(seconds):
        if seconds is None:
            return "00:00:00,000"
        whole_seconds = int(seconds)
        milliseconds = int((seconds - whole_seconds) * 1000)

        hours = whole_seconds // 3600
        minutes = (whole_seconds % 3600) // 60
        seconds = whole_seconds % 60

        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

    with open(input_path, 'r') as file:
        data = json.load(file)

    rst_string = ''
    for index, chunk in enumerate(data['chunks'], 1):
        text = chunk['text']
        start, end = chunk.get('timestamp', [None, None])
        if start is None or end is None:
            print(f"Warning: Chunk {index} has missing timestamps. Skipping...")
            continue
        start_format, end_format = format_seconds(start), format_seconds(end)
        srt_entry = f"{index}\n{start_format} --> {end_format}\n{text}\n\n"

        rst_string += srt_entry

    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(rst_string)

def count_words_str_file(rst_string):
    total_characters = len(rst_string)
    total_words = len(rst_string.split())
    return total_characters, total_words

# Function to log messages
def log_message(message):
    with open(LOG_FILE, 'a') as log_file:
        log_file.write(f"{datetime.now()} - {message}\n")

# Function to validate access key
def validate_key(key):
    return key in whitelist

def get_audio_metrics(audio_path):
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=bit_rate,sample_rate", "-of", "default=noprint_wrappers=1:nokey=1", audio_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        bit_rate, sample_rate = result.stdout.strip().split()
        return int(bit_rate) // 1000, int(sample_rate)  # Convert bit_rate to kbps
    except Exception as e:
        raise RuntimeError(f"Failed to get audio metrics: {e}")

# Function to track user activity
def track_user_activity(key, file_name, url, force_reprocess, duration, output_srt, output_json, TEMP_DIR, message, video_path, total_characters, total_words, processing_time, video_format, file_size, transcription_model, audio_bitrate, audio_sample_rate):
    entry = {
        "file": file_name,
        "url": url,
        "force_reprocess": force_reprocess,
        "duration_hours": duration,
        "timestamp": datetime.now().isoformat(),
        "total_characters": total_characters,
        "total_words": total_words,
        "output_srt": output_srt,
        "output_json": output_json,
        "temp_dir": TEMP_DIR,
        "message": message,
        "video_path": video_path,
        "processing_time_seconds": processing_time,
        "video_format": video_format,
        "file_size_bytes": file_size,
        "transcription_model": transcription_model,
        "audio_bitrate_kbps": audio_bitrate,
        "audio_sample_rate_hz": audio_sample_rate
    }

    if key not in user_activity:
        user_activity[key] = {
            "total_videos": 0,
            "total_hours": 0.0,
            "total_characters": 0,
            "total_words": 0,
            "entries": []
        }
    
    user_activity[key]["total_videos"] += 1
    user_activity[key]["total_hours"] += duration
    user_activity[key]["total_characters"] += total_characters
    user_activity[key]["total_words"] += total_words
    user_activity[key]["entries"].append(entry)
    
    save_user_activity()

# Function to get user stats
def get_user_stats(key):
    if key in user_activity:
        stats = user_activity[key]
        return (
            f"Total Videos: {stats['total_videos']}\n"
            f"Total Hours: {stats['total_hours']}\n"
            f"Total Characters: {stats['total_characters']}\n"
            f"Total Words: {stats['total_words']}"
        )
    else:
        return "No activity found for this key."

# Function to handle the Gradio interface
def transcribe_video(key, url, uploaded_file=None, force_reprocess=False, audio_format='wav'):
    if not validate_key(key):
        status_queue.put("Wrong Access Key - Check Key")
        return "", "", ""

    check_ffmpeg()  # Ensure ffmpeg is installed

    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    api_call_timestamp = datetime.now().isoformat()
    start_time = time.time()
    video_format = None
    file_size = None

    try:
        if uploaded_file is not None:
            video_path = uploaded_file
            sanitized_file_name = sanitize_filename(os.path.basename(uploaded_file))
            sanitized_path = os.path.join(TEMP_DIR, sanitized_file_name)
            shutil.copy(uploaded_file, sanitized_path)
            video_path = sanitized_path
            duration = 0  # Unable to calculate duration for uploaded files
            video_format = os.path.splitext(uploaded_file)[1][1:]
            file_size = os.path.getsize(uploaded_file)
        else:
            # Check if the URL has been processed before
            if url in processed_urls and not force_reprocess:
                json_file, srt_file = processed_urls[url]
                return "Success", json_file, srt_file

            status_queue.put("Downloading video...")
            video_path, duration = download_video(url)
            video_format = os.path.splitext(video_path)[1][1:]
            file_size = os.path.getsize(video_path)

        status_queue.put("Processing video...")
        json_file, srt_file = process_video(video_path, force_reprocess)
        processing_time = time.time() - start_time

        # Read the SRT file to get the text
        with open(srt_file, 'r', encoding='utf-8') as f:
            srt_content = f.read()

        # Calculate total characters and total words
        total_characters, total_words = count_words_str_file(srt_content)

        # Get audio metrics
        audio_path = convert_video_to_audio(video_path)
        audio_bitrate, audio_sample_rate = get_audio_metrics(audio_path)

        # Track user activity
        track_user_activity(
            key, os.path.basename(video_path), url, force_reprocess, duration / 3600.0,  # Convert duration to hours
            output_srt=srt_file, output_json=json_file, TEMP_DIR=TEMP_DIR, 
            message="Transcription successful", video_path=video_path, 
            total_characters=total_characters, total_words=total_words,
            processing_time=processing_time, video_format=video_format, 
            file_size=file_size, transcription_model="openai/whisper-large-v3",
            audio_bitrate=audio_bitrate, audio_sample_rate=audio_sample_rate
        )

        status_queue.put("Transcription successful")
        return "Success", json_file, srt_file
    except Exception as e:
        status_queue.put(f"Error: {str(e)}")
        return "", "", ""

# Function to handle video download progress
def download_progress_hook(d):
    if d['status'] == 'downloading':
        status_queue.put(f"Downloading: {d['_percent_str']} - {d['_eta_str']} remaining")
    elif d['status'] == 'finished':
        status_queue.put("Download complete")

# Function to get status updates from the queue
def get_status():
    while True:
        try:
            status = status_queue.get_nowait()
        except queue.Empty:
            status = ""
        yield status

# Gradio interface for transcription
iface = gr.Interface(
    fn=transcribe_video,
    inputs=[
        gr.Textbox(label="Enter Access Key"),
        gr.Textbox(label="Enter A Video URL"),
        gr.File(label="Upload Video File", type="filepath"),
        gr.Checkbox(label="Force Reprocess"),
        gr.Radio(label="Audio Format - Upload Files Only", choices=["wav", "mp3", "aac"], value="wav")
    ],
    outputs=[
        gr.Textbox(label="Status", interactive=False, elem_id="status"),
        gr.File(label="JSON File"),
        gr.File(label="SRT File")
    ],
    live=False,
    title="Fast LMT2 - Created by Sabian Hibbs",
    description="""Version 1.0.98 - Recent Updates:

- Access Keys: Added whitelist for known users. Tracks hours, requests, and force reprocesses.

- Status Bar: Currently buggy, will show accurate processing times soon.

- Conversion Settings: For uploaded files, use WAV as default if unsure.

- Force Reprocess: Reprocesses videos even if previously processed.

    """
)

# Gradio interface for showing user stats
stats_interface = gr.Interface(
    fn=get_user_stats,
    inputs=[gr.Textbox(label="Enter Access Key")],
    outputs=[gr.Textbox(label="User Stats")],
    live=False,
    title="User Stats",
    description="Enter your access key to view your usage statistics."
)

# Combine both interfaces
combined_interface = gr.TabbedInterface([iface, stats_interface], ["Transcribe Video", "Show User Stats"])

# Function to update status textbox
def update_status():
    status_generator = get_status()
    while True:
        status = next(status_generator)
        gr.update(value=status, elem_id="status")
        time.sleep(1)  # Adjust the sleep duration as needed

if __name__ == "__main__":
    status_thread = threading.Thread(target=update_status, daemon=True)
    status_thread.start()
    combined_interface.launch(share=True)
