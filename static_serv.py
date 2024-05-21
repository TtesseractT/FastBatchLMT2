import os
import shutil
import subprocess
import json
import gradio as gr
import yt_dlp

# Define global variables and paths
TEMP_DIR = "temp"
OUTPUT_DIR = "output"
INPUT_VIDEO_DIR = "Input-Videos"

# Check if ffmpeg is installed
def check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        raise RuntimeError("ffmpeg is not installed or not found in PATH")

# Function to download video using yt-dlp
def download_video(url):
    ydl_opts = {
        'outtmpl': os.path.join(TEMP_DIR, '%(title)s.%(ext)s'),
        'format': 'bestvideo[height<=144]+bestaudio/best',  # lowest video quality and best audio quality
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        return ydl.prepare_filename(info)

# Function to convert video to audio
def convert_video_to_audio(video_path):
    audio_path = os.path.splitext(video_path)[0] + '.wav'
    try:
        subprocess.run(
            ["ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2", audio_path],
            check=True
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg failed to convert video to audio: {e}")
    return audio_path

# Function to process the video and generate transcription
def process_video(file_path):
    file_name = os.path.basename(file_path)
    file_base = os.path.splitext(file_name)[0]
    output_json = os.path.join(TEMP_DIR, f"{file_base}.json")

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
    convert_to_srt(output_json, os.path.join(TEMP_DIR, f"{file_base}.srt"))

    # Move output files to OUTPUT_DIR
    shutil.move(output_json, os.path.join(OUTPUT_DIR, f"{file_base}.json"))
    shutil.move(os.path.join(TEMP_DIR, f"{file_base}.srt"), os.path.join(OUTPUT_DIR, f"{file_base}.srt"))

    return f"{file_base}.json", f"{file_base}.srt"

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

# Function to handle the Gradio interface
def transcribe_video(url):
    check_ffmpeg()  # Ensure ffmpeg is installed

    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    video_path = download_video(url)
    json_file, srt_file = process_video(video_path)
    
    return os.path.join(OUTPUT_DIR, json_file), os.path.join(OUTPUT_DIR, srt_file)

# Gradio interface
iface = gr.Interface(
    fn=transcribe_video,
    inputs=gr.Textbox(label="YouTube URL"),
    outputs=[gr.File(label="JSON File"), gr.File(label="SRT File")],
    live=False,
)

if __name__ == "__main__":
    iface.launch(share=True)
