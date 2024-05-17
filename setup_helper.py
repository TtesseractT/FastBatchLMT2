import os
import subprocess
import sys

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
    Set an environment variable in the system.
    """
    try:
        subprocess.run(['setx', name, value], check=True, shell=True)
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
            set_environment_variable('PATH', updated_path)
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

def install_torch_with_cuda():
    """
    Install the correct version of PyTorch with CUDA support.
    """
    try:
        print("Installing PyTorch with CUDA support...")
        subprocess.run([
            sys.executable, "-m", "pip", "install",
            "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu124"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error installing PyTorch with CUDA support: {e}")
        sys.exit(1)

def main():
    print("Setting up CUDA paths...")

    cuda_path = find_cuda_path()
    print(f"CUDA installation found at: {cuda_path}")

    cuda_bin_path = os.path.join(cuda_path, 'bin')
    cuda_nvvp_path = os.path.join(cuda_path, 'libnvvp')

    update_system_path(cuda_bin_path)
    update_system_path(cuda_nvvp_path)

    set_environment_variable('CUDA_HOME', cuda_path)

    print("CUDA paths have been successfully added to the system PATH.")
    print("Please restart your terminal or system for the changes to take effect.")

    # Uninstall any existing PyTorch installation
    uninstall_torch()

    # Install the correct version of PyTorch with CUDA support
    install_torch_with_cuda()

    # Check if PyTorch can use CUDA after setting up the environment and reinstalling PyTorch
    if is_torch_cuda_available():
        print("PyTorch is now using CUDA.")
    else:
        print("PyTorch is still not using CUDA. Please ensure the correct version of PyTorch with CUDA support is installed.")

if __name__ == "__main__":
    main()
