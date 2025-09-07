import os
import sys

# On Windows (Python 3.8+), explicitly add CUDA/cuDNN bin dirs to DLL search path
if sys.platform == "win32":
    try:
        add_dir = getattr(os, "add_dll_directory", None)
        if add_dir is not None:
            # Prefer an explicit cuDNN bin if provided
            cudnn_bin = os.environ.get("CUDNN_BIN")
            if cudnn_bin and os.path.isdir(cudnn_bin):
                add_dir(cudnn_bin)

            # Also add CUDA bin from CUDA_PATH if available
            cuda_path = os.environ.get("CUDA_PATH")
            if cuda_path:
                cuda_bin = os.path.join(cuda_path, "bin")
                if os.path.isdir(cuda_bin):
                    add_dir(cuda_bin)
    except Exception:
        # Non-fatal; continue and let imports fail if truly missing
        pass

import torch

cuda_ver = getattr(getattr(torch, "version", None), "cuda", None)
cudnn_ver = None
try:
    cudnn_ver = torch.backends.cudnn.version()
except Exception:
    pass

print("CUDA", cuda_ver, "cuDNN", cudnn_ver, "Avail", torch.cuda.is_available())
