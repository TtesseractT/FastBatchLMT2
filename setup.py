import os
import subprocess
import sys
from pathlib import Path
import argparse
import yaml
import os
import tempfile

#-------------------------------------------------------------------#
# BatchLMT2 - LOCAL                                                 #
#-------------------------------------------------------------------#
# Author: TTESSERACTT                                               #
# License: Apache License                                           #
# Version: 1.0.1                                                    #
#-------------------------------------------------------------------#


def is_torch_cuda_available():
    """
    Check if PyTorch has been compiled with CUDA enabled by moving a tensor to the GPU.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        # Try to move a tensor to the GPU to confirm CUDA is enabled
        try:
            torch.zeros(1).cuda()
            return True
        except AssertionError:
            return False
    except Exception as e:
        print(f"Error checking CUDA availability in PyTorch: {e}")
        return False

def find_cuda_version():
    """
    Find the CUDA version installed on the system.
    """
    try:
        output = subprocess.check_output("nvcc --version", shell=True, stderr=subprocess.STDOUT, text=True)
        for line in output.split('\n'):
            if 'release' in line:
                version_str = line.split('release')[-1].strip().split(',')[0]
                return float(version_str)
    except subprocess.CalledProcessError:
        print("CUDA not found. Please ensure CUDA is installed.")
        sys.exit(1)
    return None

def find_cuda_path():
    """
    Find the CUDA installation path on the system.
    """
    try:
        output = subprocess.check_output("where nvcc", shell=True, stderr=subprocess.STDOUT, text=True)
        cuda_bin_path = os.path.dirname(output.strip())
        cuda_path = os.path.dirname(cuda_bin_path)
        return cuda_path
    except subprocess.CalledProcessError:
        print("CUDA not found. Please ensure CUDA is installed.")
        sys.exit(1)

def set_environment_variable(name, value):
    """
    Set an environment variable in the system and in the current process.
    """
    try:
        os.environ[name] = value  # Set it for the current process
        subprocess.run(['setx', name, value], check=True, shell=True)  # Set it for future processes
    except subprocess.CalledProcessError as e:
        print(f"Error setting environment variable {name}. Error: {e}")
        sys.exit(1)

def update_system_path(new_path):
    """
    Add a new path to the system PATH environment variable.
    """
    try:
        current_path = os.environ['PATH']
        if new_path not in current_path.split(';'):
            updated_path = f"{current_path};{new_path}"
            os.environ['PATH'] = updated_path  # Set it for the current process
            set_environment_variable('PATH', updated_path)  # Set it for future processes
        else:
            print(f"{new_path} is already in the PATH.")
    except KeyError:
        print("Error retrieving the PATH environment variable.")
        sys.exit(1)

def uninstall_torch():
    """
    Uninstall existing PyTorch installation.
    """
    try:
        print("Uninstalling existing PyTorch...")
        subprocess.run([
            sys.executable, "-m", "pip", "uninstall", "-y", "torch", "torchvision", "torchaudio"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error uninstalling PyTorch: {e}")
        sys.exit(1)

def install_torch_with_cuda(cuda_version):
    """
    Install the correct version of PyTorch with CUDA support based on the CUDA version.
    """
    try:
        if cuda_version <= 11.8:
            torch_index_url = "https://download.pytorch.org/whl/cu118"
        elif cuda_version <= 12.1:
            torch_index_url = "https://download.pytorch.org/whl/cu121"
        else:
            torch_index_url = "https://download.pytorch.org/whl/cu121"

        print(f"Installing PyTorch with CUDA support from {torch_index_url}...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio", "--index-url", torch_index_url
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error installing PyTorch with CUDA support: {e}")
        sys.exit(1)

def verify_paths(cuda_bin_path, cuda_nvvp_path, cuda_home):
    """
    Verify that the necessary CUDA paths are correctly set.
    """
    try:
        current_path = os.environ['PATH']
        if cuda_bin_path not in current_path.split(';'):
            raise EnvironmentError(f"{cuda_bin_path} is not in the system PATH.")
        if cuda_nvvp_path not in current_path.split(';'):
            raise EnvironmentError(f"{cuda_nvvp_path} is not in the system PATH.")
        
        if os.environ.get('CUDA_HOME') != cuda_home:
            raise EnvironmentError(f"CUDA_HOME is not set correctly. Expected: {cuda_home}, Found: {os.environ.get('CUDA_HOME')}")

        print("All necessary CUDA paths are correctly set.")
    except EnvironmentError as e:
        print(f"Path verification error: {e}")
        sys.exit(1)

def Overall_Process():
    print("Setting up CUDA paths...")

    cuda_path = find_cuda_path()
    print(f"CUDA installation found at: {cuda_path}")

    cuda_bin_path = os.path.join(cuda_path, 'bin')
    cuda_nvvp_path = os.path.join(cuda_path, 'libnvvp')

    update_system_path(cuda_bin_path)
    update_system_path(cuda_nvvp_path)

    set_environment_variable('CUDA_HOME', cuda_path)

    print("CUDA paths have been successfully added to the system PATH.")

    # Find CUDA version
    cuda_version = find_cuda_version()
    print(f"Detected CUDA version: {cuda_version}")

    # Uninstall any existing PyTorch installation
    uninstall_torch()

    # Install the correct version of PyTorch with CUDA support
    install_torch_with_cuda(cuda_version)

    # Verify paths after installation
    verify_paths(cuda_bin_path, cuda_nvvp_path, cuda_path)

    print("Please restart your terminal or system for the changes to take effect.")

    # Check if PyTorch can use CUDA after setting up the environment and reinstalling PyTorch
    if is_torch_cuda_available():
        print("PyTorch is now using CUDA.")
    else:
        print("PyTorch is still not using CUDA. Please ensure the correct version of PyTorch with CUDA support is installed.")

def main():
    print('Checking Deps')
    subprocess.run(['whisper', '--version'], shell=True)
    print("Installing Whisper")
    subprocess.run(['pip', 'install', 'git+https://github.com/openai/whisper.git'], shell=True)
    subprocess.run(['pip', 'install', '--upgrade', '--no-deps', '--force-reinstall', 'git+https://github.com/openai/whisper.git'], shell=True)

    print("Checking Dependencies...")
    try: 
        # Check if pipx is installed
        import pipx
    except:
        subprocess.run(['pip', 'install', 'pipx'], shell=True)
    # install pynvml
    
    try:
        import pynvml
    except:
        subprocess.run(['pip', 'install', 'pynvml'], shell=True)
    
    try:
        subprocess.run(['insanely-fast-whisper', '--version'], shell=True)
    except:
        print("Installing Insanely Fast Whisper")
        subprocess.run(['pipx', 'install', 'insanely-fast-whisper'], shell=True)

    print("Checking Installation...")
    Overall_Process()

if __name__ == "__main__":
    # Create the directories if they don't exist
    if not os.path.exists('Input-Videos'):
        os.mkdir('Input-Videos')
    if not os.path.exists('Videos'):
        os.mkdir('Videos')
    print("Checking Installation...")
    main()