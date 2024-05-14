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


def init_gpu():
    """ Initializes the NVIDIA Management Library (NVML). """
    pynvml.nvmlInit()

def shutdown_gpu():
    """ Shuts down the NVIDIA Management Library (NVML). """
    pynvml.nvmlShutdown()

def get_gpu_memory_info():
    """
    Retrieves the total and free GPU memory information for all GPUs.
    Returns:
        dict: A dictionary with GPU indices as keys and tuples of (total, free) memory in bytes as values.
    """
    device_count = pynvml.nvmlDeviceGetCount()
    gpu_memory_info = {}
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_memory_info[i] = (info.total, info.free)
    return gpu_memory_info

def find_available_gpu(minimum_mem_required):
    """
    Finds the first available GPU with at least 'minimum_mem_required' bytes of free memory.

    Args:
        minimum_mem_required (int): The minimum amount of free memory required in bytes.

    Returns:
        int: The index of the GPU if available, otherwise None.
    """
    gpu_memory_info = get_gpu_memory_info()
    for gpu_index, (total, free) in gpu_memory_info.items():
        if free >= minimum_mem_required:
            return gpu_index
    return None

def process_file(file_to_process, video_folder_name, gpu_id):
    """
    Processes a video file using the specified GPU.
    Args:
        file_to_process (str): The name of the file to be processed.
        video_folder_name (str): The name of the folder to store the processed video and related files.
        gpu_id (int): The GPU ID to use for processing.
    """
    try:
        print(f"Attempting to process {file_to_process} on GPU {gpu_id}")
        shutil.move(os.path.join('Input-Videos', file_to_process), file_to_process)
        subprocess.run(f'insanely-fast-whisper --file-name "{file_to_process}" --model-name openai/whisper-large-v3 --task transcribe --language en --device-id {gpu_id} --transcript-path "{os.path.splitext(file_to_process)[0]}.json', shell=True)

        new_folder_path = os.path.join('Videos', video_folder_name)
        os.mkdir(new_folder_path)
        shutil.move(file_to_process, new_folder_path)
        for filename in os.listdir('.'):
            if filename.startswith(os.path.splitext(file_to_process)[0]):
                shutil.move(filename, new_folder_path)

    except Exception as e:
        print(f"Processing failed with error: {e}")
        print("Reversing the file operations...")
        if os.path.exists(os.path.join('Videos', video_folder_name, file_to_process)):
            shutil.move(os.path.join('Videos', video_folder_name, file_to_process), '.')
        if os.path.exists(os.path.join('Videos', video_folder_name)):
            shutil.rmtree(os.path.join('Videos', video_folder_name))
        move_and_clear_videos()

def worker(file_queue):
    """
    Worker function to process files from a queue.

    Continuously retrieves files from the queue and processes them until the queue is empty.

    Args:
        file_queue (queue.Queue): A queue containing files to be processed. Each item in the queue
                                  should be a tuple where the first element is the file name and 
                                  the second element is an identifier used to create a folder name.
    
    Raises:
        queue.Empty: If the queue is empty when attempting to retrieve a file, the function breaks the loop.
    """
    while not file_queue.empty():
        try:
            file_to_process = file_queue.get_nowait()
        except queue.Empty:
            break

        video_folder_name = f'Video - {file_to_process[1]}'
        process_file(file_to_process[0], video_folder_name)
        file_queue.task_done()

def process_files_LMT2_batch():
    """
    Processes video files in batches, utilizing available GPU memory to determine GPU assignment.
    """
    init_gpu()  # Ensure NVML is initialized
    minimum_mem_required = 5.5 * 1024**3  # Minimum memory required per GPU to process a file.
    input_dir = 'Input-Videos'
    files_to_process = os.listdir(input_dir)
    
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(files_to_process)) as executor:
            futures = []
            for file_to_process in files_to_process:
                gpu_id = find_available_gpu(minimum_mem_required)
                if gpu_id is not None:
                    video_folder_name = f'Video-{os.path.splitext(file_to_process)[0]}'
                    futures.append(executor.submit(process_file, file_to_process, video_folder_name, gpu_id))
                else:
                    print(f"No available GPU with enough memory to process {file_to_process}")
            concurrent.futures.wait(futures)
    finally:
        shutdown_gpu()  # Clean up NVML


def cleanup_filenames():
    """
    Renames subdirectories in the 'Videos' folder based on the largest file within each subdirectory.

    This function iterates over all subdirectories in the 'Videos' folder, finds the largest file in each
    subdirectory, and renames the subdirectory to match the name (without extension) of the largest file.

    Raises:
        Exception: If there is any issue with reading directories or renaming files, an exception is raised.
    """
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
    """
    Moves all .mp4 files from the current directory to the 'Input-Videos' directory and clears specified file types 
    and the 'Videos' directory.

    This function performs the following steps:
    1. Creates the 'Input-Videos' directory if it does not exist.
    2. Moves all .mp4 files from the current directory to the 'Input-Videos' directory.
    3. Removes specified file types from the current directory.
    4. Clears all files and subdirectories within the 'Videos' directory, then removes the 'Videos' directory itself.

    Raises:
        Exception: If there is any issue with file operations, an exception will be raised.
    """
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
    """
    Formats a duration given in seconds into a string with the format "HH:MM:SS,mmm".

    If the input is None, returns a default string "00:00:00,000".

    Args:
        seconds (float or None): The duration in seconds to format. If None, returns the default time string.

    Returns:
        str: The formatted time string in the format "HH:MM:SS,mmm".
    """
    if seconds is None:
        return "00:00:00,000"
    whole_seconds = int(seconds)
    milliseconds = int((seconds - whole_seconds) * 1000)

    hours = whole_seconds // 3600
    minutes = (whole_seconds % 3600) // 60
    seconds = whole_seconds % 60

    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

def convert_to_srt(input_path, output_path, verbose):
    """
    Formats a duration given in seconds into a string with the format "HH:MM:SS,mmm".

    If the input is None, returns a default string "00:00:00,000".

    Args:
        seconds (float or None): The duration in seconds to format. If None, returns the default time string.

    Returns:
        str: The formatted time string in the format "HH:MM:SS,mmm".
    """
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
    """
    Processes all JSON files in the 'Videos' directory by converting them to SRT subtitle files.

    This function searches through all subdirectories in the 'Videos' folder, converts each JSON file
    to an SRT file using the `convert_to_srt` function, and saves the SRT files in the same subdirectory.

    Args:
        verbose (bool, optional): If True, prints each SRT entry during the conversion process.

    Raises:
        FileNotFoundError: If the 'Videos' directory does not exist.
        json.JSONDecodeError: If there is an error decoding the JSON file during conversion.
    """
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

    # Example Useage
    # python fast_batch.py
