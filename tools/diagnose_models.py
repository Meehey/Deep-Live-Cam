import os
import sys
import json
import onnxruntime as ort

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from modules import globals as G  # type: ignore
from modules import core  # type: ignore
from modules.face_analyser import get_face_analyser  # type: ignore
from modules.processors.frame.face_swapper import (  # type: ignore
    pre_check as swapper_pre_check,
    get_face_swapper,
)


def main():
    # Run diagnostics without touching the GUI layer
    G.headless = True
    report = {"available_providers": ort.get_available_providers()}

    # Choose providers: prefer CUDA, then DML/ROCM/CoreML if present, else CPU
    want = ["cuda", "dml", "rocm", "coreml", "cpu"]
    try:
        providers = core.decode_execution_providers(want)
    except Exception:
        providers = []
    if not providers:
        providers = ["CPUExecutionProvider"]
    G.execution_providers = providers
    report["selected_providers"] = providers

    # Check files
    root = G.MODELS_DIR
    report["models_root"] = root
    buffalo_dir = os.path.join(root, "models", "buffalo_l")
    report["buffalo_l_dir"] = buffalo_dir
    expected_blf = ["det_10g.onnx", "2d106det.onnx", "w600k_r50.onnx"]
    report["buffalo_l_files"] = {
        f: os.path.exists(os.path.join(buffalo_dir, f)) for f in expected_blf
    }

    # inswapper
    fp16 = "CUDAExecutionProvider" in providers
    inswapper = os.path.join(
        root, "inswapper_128_fp16.onnx" if fp16 else "inswapper_128.onnx"
    )
    report["inswapper_model"] = {"path": inswapper, "exists": os.path.exists(inswapper)}

    # Try initializations
    init = {"face_analyser": None, "face_swapper": None}
    errors = {}

    try:
        get_face_analyser()
        # InsightFace doesn't expose internal sessions easily; note it's created
        init["face_analyser"] = True
    except Exception as e:
        init["face_analyser"] = False
        errors["face_analyser"] = f"{type(e).__name__}: {e}"

    try:
        swapper_pre_check()
        get_face_swapper()
        init["face_swapper"] = True
    except Exception as e:
        init["face_swapper"] = False
        errors["face_swapper"] = f"{type(e).__name__}: {e}"

    report["init_ok"] = init
    if errors:
        report["errors"] = errors

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
