#-------------------------------------------------------------------#
# BatchLMT2 - LOCAL                                                 #
#-------------------------------------------------------------------#
# Author: TTESSERACTT                                               #
# License: Apache License                                           #
# Version: 1.0.1                                                    #
#-------------------------------------------------------------------#


import concurrent.futures
import subprocess
import argparse
import platform
import shutil
import pynvml
import signal
import queue
import time
import json
import sys
import os


def get_gpu_memory_info():
    pynvml.nvmlInit()
    gpu_info = []
    for i in range(pynvml.nvmlDeviceGetCount()):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_info.append((info.total, info.free))
    pynvml.nvmlShutdown()
    return gpu_info

def process_file(file_to_process, video_folder_name, gpu_id):
    try:
        # Move the file to the processing directory
        shutil.move(os.path.join('Input-Videos', file_to_process), file_to_process)
        print(f"Processing file: {file_to_process}")
        filenamestatic = os.path.splitext(file_to_process)[0]
        print(filenamestatic)

        subprocess.run(f'insanely-fast-whisper --file-name "{file_to_process}" --model-name openai/whisper-large-v3 --task transcribe --language en --device-id {gpu_id} --transcript-path "{filenamestatic}".json', shell=True)

        # Create a new directory for the processed video and move all related files
        new_folder_path = os.path.join('Videos', video_folder_name)
        os.mkdir(new_folder_path)

        # Move the original file and all related output files
        shutil.move(file_to_process, new_folder_path)
        output_file_base = os.path.splitext(file_to_process)[0]
        for filename in os.listdir('.'):
            if filename.startswith(output_file_base):
                shutil.move(filename, new_folder_path)

    except Exception as e:
        move_and_clear_videos()

def worker(file_queue):
    while not file_queue.empty():
        try:
            file_to_process = file_queue.get_nowait()
        except queue.Empty:
            break

        video_folder_name = f'Video - {file_to_process[1]}'
        gpu_info = get_gpu_memory_info()
        available_gpus = [(gpu_id, free_mem) for gpu_id, (total_mem, free_mem) in enumerate(gpu_info) if free_mem > 11 * 1024**3]

        if available_gpus:
            gpu_id, _ = max(available_gpus, key=lambda x: x[1])
            process_file(file_to_process[0], video_folder_name, gpu_id)
        else:
            print("No available GPU found with sufficient memory.")

        file_queue.task_done()

def process_files_LMT2_batch():
    gpu_info = get_gpu_memory_info()
    vram_per_process = 11 * 1024**3  # Assuming each process requires around 11 GB of VRAM
    input_dir = 'Input-Videos'
    files_to_process = os.listdir(input_dir)
    num_files = len(files_to_process)
    file_queue = queue.Queue()
    for i, file_to_process in enumerate(files_to_process, 1):
        file_queue.put((file_to_process, i))

    max_workers = len(gpu_info)  # Set max workers to the number of GPUs

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(worker, file_queue) for _ in range(max_workers)]
        concurrent.futures.wait(futures)


def cleanup_filenames():
    videos_folder = "./Videos"

    for subdir in os.listdir(videos_folder):
        subdir_path = os.path.join(videos_folder, subdir)
        if os.path.isdir(subdir_path):
            files = os.listdir(subdir_path)
            if files:
                largest_file = max(files, key=lambda f: os.path.getsize(os.path.join(subdir_path, f)))
                new_name = os.path.splitext(largest_file)[0]
                os.rename(subdir_path, os.path.join(videos_folder, new_name))

def move_and_clear_videos():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    target_directory = os.path.join(current_directory, 'Input-Videos')
    videos_directory = os.path.join(current_directory, 'Videos')
    extensions_to_remove = ['.json', '.srt', '.tsv', '.txt', '.vtt']

    # Create 'Input-Videos' directory if it doesn't exist
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    # Find and move all .mp4 files
    for root, dirs, files in os.walk(current_directory):
        for file in files:
            if file.endswith('.mp4'):
                source_path = os.path.join(root, file)
                destination_path = os.path.join(target_directory, file)
                print(f"Moving {source_path} to {destination_path}")
                shutil.move(source_path, destination_path)
        # Only process the top directory (current directory)
        break

    # Remove specified files in the current directory
    for file in os.listdir(current_directory):
        if any(file.endswith(ext) for ext in extensions_to_remove):
            file_path = os.path.join(current_directory, file)
            print(f"Removing file {file_path}")
            os.remove(file_path)

    # Clear the 'Videos' directory
    if os.path.exists(videos_directory):
        for root, dirs, files in os.walk(videos_directory, topdown=False):
            for file in files:
                file_path = os.path.join(root, file)
                print(f"Removing file {file_path}")
                os.remove(file_path)
            for dir in dirs:
                dir_path = os.path.join(root, dir)
                print(f"Removing directory {dir_path}")
                os.rmdir(dir_path)
        os.rmdir(videos_directory)  # Remove the 'Videos' directory itself

def format_seconds(seconds):
    if seconds is None:
        return "00:00:00,000"
    whole_seconds = int(seconds)
    milliseconds = int((seconds - whole_seconds) * 1000)

    hours = whole_seconds // 3600
    minutes = (whole_seconds % 3600) // 60
    seconds = whole_seconds % 60

    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

def convert_to_srt(input_path, output_path, verbose):
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

        if verbose:
            print(srt_entry)

        rst_string += srt_entry

    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(rst_string)

def process_json_files_in_videos(verbose=False):
    videos_folder = "./Videos"
    
    for subdir in os.listdir(videos_folder):
        subdir_path = os.path.join(videos_folder, subdir)
        if os.path.isdir(subdir_path):
            for file in os.listdir(subdir_path):
                if file.endswith('.json'):
                    json_path = os.path.join(subdir_path, file)
                    srt_path = os.path.join(subdir_path, os.path.splitext(file)[0] + '.srt')
                    convert_to_srt(json_path, srt_path, verbose)
                    print(f"Converted {json_path} to {srt_path}")

if __name__ == '__main__':
    try:
        start_time = time.time()  # Record the start time
        
        process_files_LMT2_batch()
        cleanup_filenames()
        process_json_files_in_videos()
        
        end_time = time.time()  # Record the end time
        elapsed_time = end_time - start_time  # Calculate the elapsed time
        
        print(f"Script completed in {elapsed_time:.2f} seconds")
    except Exception as e:
        move_and_clear_videos() # Basic cleanup on error

