#-------------------------------------------------------------------#
# BatchLMT2 - LOCAL                                                 #
#-------------------------------------------------------------------#
# Author: TTESSERACTT                                               #
# License: Apache License                                           #
# Version: 1.0.1                                                    #
#-------------------------------------------------------------------#


import os
import shutil
import argparse


def find_and_move_mp4_files(input_directory, target_directory):
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    for root, dirs, files in os.walk(input_directory):
        for file in files:
            if file.endswith('.mp4'):
                source_path = os.path.join(root, file)
                destination_path = os.path.join(target_directory, file)
                print(f"Moving {source_path} to {destination_path}")
                shutil.move(source_path, destination_path)

def clear_directory(directory):
    for root, dirs, files in os.walk(directory, topdown=False):
        for file in files:
            file_path = os.path.join(root, file)
            print(f"Removing file {file_path}")
            os.remove(file_path)
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            print(f"Removing directory {dir_path}")
            os.rmdir(dir_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Move .mp4 files to 'Input-Videos' and clear 'Videos' directory.")
    parser.add_argument('-i', '--input', type=str, required=True, help="Path to the input directory")

    args = parser.parse_args()
    input_directory = args.input
    target_directory = 'Input-Videos'
    videos_directory = 'Videos'

    find_and_move_mp4_files(input_directory, target_directory)
    clear_directory(videos_directory)