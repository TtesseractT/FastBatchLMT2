import gradio as gr
import yt_dlp
import subprocess
import os
import tempfile
import json

def format_seconds(seconds):
    if seconds is None:
        return "00:00:00,000"
    whole_seconds = int(seconds)
    milliseconds = int((seconds - whole_seconds) * 1000)
    hours = whole_seconds // 3600
    minutes = (whole_seconds % 3600) // 60
    seconds = whole_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

def convert_to_srt(input_path, output_path, verbose=False):
    with open(input_path, 'r') as file:
        data = json.load(file)
    srt_string = ''
    for index, chunk in enumerate(data['chunks'], 1):
        text = chunk['text']
        start, end = chunk.get('timestamp', [None, None])
        if start is None or end is None:
            if verbose:
                print(f"Warning: Chunk {index} has missing timestamps. Skipping...")
            continue
        start_format, end_format = format_seconds(start), format_seconds(end)
        srt_entry = f"{index}\n{start_format} --> {end_format}\n{text}\n\n"
        if verbose:
            print(srt_entry)
        srt_string += srt_entry
    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(srt_string)

def download_and_process(url):
    temp_dir = tempfile.mkdtemp()
    video_path = os.path.join(temp_dir, "video.mp4")
    ydl_opts = {
    'format': 'worstvideo+bestaudio/best',  # Combines the worst quality video with the best quality audio
    'outtmpl': video_path,
    'quiet': True,
    'merge_output_format': 'mp4',  # Ensure the output is mp4
    'postprocessors': [{
        'key': 'FFmpegVideoConvertor',
        'preferedformat': 'mp4',  # Explicitly convert to mp4 if necessary
    }]
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    
    json_path = os.path.join(temp_dir, "transcription.json")
    srt_path = os.path.join(temp_dir, "transcription.srt")
    
    # Transcribe using insanely-fast-whisper
    cmd = f"insanely-fast-whisper --file-name '{video_path}' --model-name openai/whisper-large-v3 --task transcribe --language en --device-id 0 --transcript-path '{json_path}'"
    subprocess.run(cmd, shell=True)
    
    # Convert JSON to SRT
    convert_to_srt(json_path, srt_path, verbose=True)
    
    return temp_dir, json_path, srt_path

def gradio_interface():
    with gr.Blocks() as demo:
        gr.Markdown("### Video Transcription Demo")
        with gr.Row():
            url_input = gr.Textbox(label="Enter Video URL")
            submit_button = gr.Button("Submit")
        outputs = gr.File(label="Download JSON"), gr.File(label="Download SRT")
        
        submit_button.click(fn=download_and_process, inputs=[url_input], outputs=[outputs[0], outputs[1]])

    return demo

app = gradio_interface()
app.launch(share=True)

