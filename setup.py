#-------------------------------------------------------------------#
# BatchLMT2 - LOCAL                                                 #
#-------------------------------------------------------------------#
# Author: TTESSERACTT                                               #
# License: Apache License                                           #
# Version: 1.0.1                                                    #
#-------------------------------------------------------------------#

import os
import subprocess
import sys
from pathlib import Path

try:
    print('Checking Deps')
    subprocess.run(['whisper', '--version'], shell=True)
except:
    subprocess.run(['pip', 'install', 'git+https://github.com/openai/whisper.git'])
    subprocess.run(['pip', 'install', '--upgrade', '--no-deps', '--force-reinstall', 'git+https://github.com/openai/whisper.git'])
    subprocess.run(['pip', 'install', 'torch', 'torchvision', 'torchaudio', '--index-url', 'https://download.pytorch.org/whl/cu118'])

    print("Checking Dependencies...")
    subprocess.run(['pip', 'install', 'pipx'], shell=True)

    # install pynvml
    subprocess.run(['pip', 'install', 'pynvml'], shell=True)

    try:
        subprocess.run(['insanely-fast-whisper', '--version'], shell=True)
    except:
        # Step 1: Check and add to PATH if necessary
        home_dir = Path.home()
        local_bin_path = home_dir / '.local' / 'bin'
        current_path = os.environ.get('PATH', '')

        if str(local_bin_path) not in current_path.split(os.pathsep):
            print(f"Adding {local_bin_path} to PATH")

            # Add the path to the user's environment variable
            command = f'setx PATH "%PATH%;{local_bin_path}"'
            subprocess.run(command, shell=True)
            
            # Update the current script's PATH
            os.environ['PATH'] += os.pathsep + str(local_bin_path)
            print(f"Path updated: {os.environ['PATH']}")
        else:
            print(f"{local_bin_path} is already in PATH")

        # Step 2: Ensure pipx is installed
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', '--user', 'pipx'], check=True)
        except subprocess.CalledProcessError as e:
            print(f"An error occurred during pipx installation: {e}")
            sys.exit(1)

        # Step 3: Ensure PATH for pipx
        try:
            subprocess.run(['pipx', 'ensurepath'], check=True)
        except subprocess.CalledProcessError as e:
            print(f"An error occurred during pipx ensurepath: {e}")
            sys.exit(1)

        # Step 4: Install or reinstall insanely-fast-whisper
        try:
            subprocess.run(['pipx', 'install', 'insanely-fast-whisper', '--force', '--pip-args=--ignore-requires-python'], check=True)
        except subprocess.CalledProcessError as e:
            print(f"An error occurred during insanely-fast-whisper installation: {e}")
            sys.exit(1)

        # Step 5: Verify installation
        try:
            result = subprocess.run(['insanely-fast-whisper', '--version'], check=True, capture_output=True, text=True)
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"An error occurred during insanely-fast-whisper verification: {e}")
            sys.exit(1)


# Create the directories if they don't exist
if not os.path.exists('Input-Videos'):
    os.mkdir('Input-Videos')
if not os.path.exists('Videos'):
    os.mkdir('Videos')
