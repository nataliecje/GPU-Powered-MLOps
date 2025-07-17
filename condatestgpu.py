import torch

print("PyTorch Version:", torch.__version__)

# Check for MPS (Metal Performance Shaders) availability
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    print(f"MPS (GPU) is available: {mps_device}")
    print(f"  MPS is built: {torch.backends.mps.is_built()}")

    # Try to put a tensor on the MPS device
    try:
        x = torch.ones(5, device=mps_device)
        print(f"  Tensor on MPS device: {x}")
        print(f"  Tensor device: {x.device}")
    except Exception as e:
        print(f"  Could not place tensor on MPS device: {e}")
else:
    print("MPS (GPU) is NOT available.")
    print("PyTorch will run on CPU.")

# Fallback to check for CUDA (not applicable for M2, but good for general check)
if torch.cuda.is_available():
    print("CUDA (NVIDIA GPU) is available: Yes (this is unexpected on M2 unless you have an eGPU)")
    print(f"  CUDA device count: {torch.cuda.device_count()}")
    print(f"  CUDA device name: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA (NVIDIA GPU) is NOT available (expected on M2 Mac).")