"""Record local software, platform, and non-training tensor/import checks."""

import importlib
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path


def main() -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(Path("work/matplotlib").resolve()))
    for module in ("numpy", "sklearn", "xgboost", "torch", "matplotlib", "pytest"):
        importlib.import_module(module)
    import torch

    tensor = torch.tensor([1.0, 2.0], device="cpu")
    assert torch.sum(tensor).item() == 3.0
    mps_available = torch.backends.mps.is_available()
    mps_check = "not_available_in_current_execution_context"
    if mps_available:
        assert torch.sum(tensor.to("mps")).cpu().item() == 3.0
        mps_check = "passed_simple_tensor_sum"
    packages = (
        "numpy",
        "scikit-learn",
        "scipy",
        "xgboost",
        "torch",
        "matplotlib",
        "pytest",
        "ruff",
        "setuptools",
        "pip",
    )
    result = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "processor_reported": platform.processor(),
        "logical_cpu_count_reported": os.cpu_count(),
        "package_versions": {name: importlib.metadata.version(name) for name in packages},
        "cpu_tensor_check": "passed",
        "mps_built": torch.backends.mps.is_built(),
        "mps_available_in_execution_context": mps_available,
        "mps_tensor_check": mps_check,
        "cuda_available": torch.cuda.is_available(),
        "models_trained": 0,
        "note": "Device availability reflects this sandbox; no training runtime measured.",
    }
    path = Path("results/task0001/environment.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
