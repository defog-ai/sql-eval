import requests
from uuid import uuid4
from datetime import datetime
import os
import hashlib

# get the GPU name this is running on
def get_gpu_name():
    """
    Get the GPU name this is running on.
    """
    # Get the GPU name
    try:
        gpu_name = os.popen("nvidia-smi --query-gpu=gpu_name --format=csv,noheader").read()
    except:
        gpu_name = "No GPU found"
    # Return the GPU name
    return gpu_name

def get_gpu_memory():
    """
    Get the GPU memory this is running on.
    """
    # Get the GPU memory
    try:
        gpu_memory = os.popen("nvidia-smi --query-gpu=memory.total --format=csv,noheader").read()
    except:
        gpu_memory = "No GPU found"
    # Return the GPU memory
    return gpu_memory

def get_gpu_driver_version():
    """
    Get the GPU driver version this is running on.
    """
    # Get the GPU driver version
    try:
        gpu_driver_version = os.popen("nvidia-smi --query-gpu=driver_version --format=csv,noheader").read()
    except:
        gpu_driver_version = "No GPU found"
    # Return the GPU driver version
    return gpu_driver_version

def get_gpu_cuda_version():
    """
    Get the GPU CUDA version this is running on.
    """
    # Get the GPU CUDA version
    try:
        gpu_cuda_version = os.popen("nvcc --version").read()
    except:
        gpu_cuda_version = "No GPU found"
    # Return the GPU CUDA version
    return gpu_cuda_version

def num_gpus():
    """
    Get the number of GPUs this is running on.
    """
    # Get the number of GPUs
    try:
        num_gpus = os.popen("nvidia-smi --query-gpu=count --format=csv,noheader").read()
    except:
        num_gpus = "No GPU found"
    # Return the number of GPUs
    return num_gpus

def upload_results(
    results: list,
    url: str,
    runner_type: str,
    prompt: str,
    model: str,
    num_beams: int = None,
    db_type: str = "postgres",
):
    """
    Uploads results to a server.
    """
    # Create a unique id for the request
    run_id = uuid4()

    # Create a unique id for the prompt, based on a hash of the prompt
    prompt_id = hashlib.md5(prompt.encode()).hexdigest()
    
    # Create a dictionary with the request id and the results
    data = {
        "run_id": run_id,
        "results": results,
        "timestamp": datetime.now().isoformat(),
        "runner_type": runner_type,
        "prompt": prompt,
        "prompt_id": prompt_id,
        "model": model,
        "num_beams": num_beams,
        "db_type": db_type,
        "gpu_name": get_gpu_name(),
        "gpu_memory": get_gpu_memory(),
        "gpu_driver_version": get_gpu_driver_version(),
        "gpu_cuda_version": get_gpu_cuda_version(),
        "num_gpus": num_gpus(),
    }
    # Send the data to the server
    response = requests.post(url, json=data)
    # Return the response
    return response