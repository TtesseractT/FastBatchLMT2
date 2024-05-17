#-------------------------------------------------------------------#
# BatchLMT2 - LOCAL                                                 #
#-------------------------------------------------------------------#
# Author: TTESSERACTT                                               #
# License: Apache License                                           #
# Version: 1.0.1                                                    #
#-------------------------------------------------------------------#


import os
import subprocess

try:
    print('Checking Deps')
    subprocess.run('whisper', '--version', shell=True)
except:
    subprocess.run('pip', 'install', 'git+https://github.com/openai/whisper.git')
    subprocess.run('pip', 'install', '--upgrade', '--no-deps', '--force-reinstall', 'git+https://github.com/openai/whisper.git')
    subprocess.run('pip', 'install', 'torch', 'torchvision', 'torchaudio', '--index-url', 'https://download.pytorch.org/whl/cu118')

# Check if Pipx is installed.
try:
    print("Checking pipx Install")
    subprocess.run('python', 'pipx', '--version')
except:
    print("pipx not installed, installing")
    subprocess.run('pip', 'install', 'pipx')

subprocess.run(f'pipx install insanely-fast-whisper', shell=True)
subprocess.run(f'pipx install insanely-fast-whisper --force --pip-args="--ignore-requires-python"', shell=True)

# Create the directories if they don't exist
if not os.path.exists('Input-Videos'):
    os.mkdir('Input-Videos')
if not os.path.exists('Videos'):
    os.mkdir('Videos')