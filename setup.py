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
    
    """
    pip install git+https://github.com/openai/whisper.git
    pip install --upgrade --no-deps --force-reinstall git+https://github.com/openai/whisper.git
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    
    """
    subprocess.run(['pip', 'install', 'git+https://github.com/openai/whisper.git'], shell=True)
    subprocess.run(['pip', 'install', '--upgrade', '--no-deps', '--force-reinstall', 'git+https://github.com/openai/whisper.git'], shell=True)
    subprocess.run(['pip', 'install', 'torch', 'torchvision', 'torchaudio', '--index-url', 'https://download.pytorch.org/whl/cu118'], shell=True)

# Check if Pipx is installed.

print("Checking Dependencies...")
subprocess.run(['pip', 'install', 'pipx'], shell=True)

# install pynvml
subprocess.run(['pip', 'install', 'pynvml'], shell=True)
print("Dependencies Installed")

print("Installing Insanely Fast Whisper")
subprocess.run(['pipx', 'install', 'insanely-fast-whisper'], shell=True)
#subprocess.run(['pipx', 'install', 'insanely-fast-whisper', '--force', '--pip-args="--ignore-requires-python"'], shell=True)

print("Verifying Installation")
subprocess.run([sys.executable, '-m', 'pip', 'install', '--user', 'pipx'], check=True)
subprocess.run(['pipx', 'ensurepath'], check=True)
subprocess.run(['$env:PATH'], shell=True)



# Create the directories if they don't exist
if not os.path.exists('Input-Videos'):
    os.mkdir('Input-Videos')
if not os.path.exists('Videos'):
    os.mkdir('Videos')
