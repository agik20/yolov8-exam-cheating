"""Generic platform environment capture for benchmark reports.

Returns a plain dict suitable for embedding in benchmark JSON.
No NVIDIA/Jetson-specific code in the default path.
Optional Jetson fields are only filled when the corresponding
system files are present; missing fields are set to None.
"""
import os
import platform
import socket
import sys
import subprocess
from typing import Dict, Optional


def _run(cmd: list, timeout=5) -> Optional[str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip() or None
    except Exception:
        return None


def _torch_info() -> Dict:
    result = {"torch_version": None, "torchvision_version": None,
              "cuda_available": False, "cuda_version": None,
              "gpu_name": None, "gpu_memory_mb": None}
    try:
        import torch
        result["torch_version"] = torch.__version__
        result["cuda_available"] = torch.cuda.is_available()
        if result["cuda_available"]:
            result["cuda_version"] = torch.version.cuda
            try:
                props = torch.cuda.get_device_properties(0)
                result["gpu_name"] = props.name
                result["gpu_memory_mb"] = int(props.total_memory / (1024 * 1024))
            except Exception:
                pass
    except ImportError:
        pass
    try:
        import torchvision
        result["torchvision_version"] = torchvision.__version__
    except ImportError:
        pass
    return result


def _cpu_info() -> Optional[str]:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    out = _run(["wmic", "cpu", "get", "name"], timeout=8)
    if out:
        return out.split("\n")[0].strip() or None
    return None


def _ram_info() -> Optional[str]:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return f"{round(kb / (1024 * 1024), 1)} GB total"
    except Exception:
        pass
    return None


def _jetson_telemetry() -> Dict:
    """Optional Jetson-specific fields.  Returns {} when not on Jetson."""
    result: Dict = {}

    model_path = "/proc/device-tree/model"
    try:
        with open(model_path, "r") as f:
            result["jetson_model"] = f.read().strip()
    except Exception:
        return {}

    try:
        import glob
        for t in sorted(glob.glob("/sys/class/thermal/thermal_zone*/temp")):
            with open(t, "r") as f:
                raw = int(f.read().strip())
                result[f"jetson_temp_{os.path.basename(os.path.dirname(t))}"] = raw / 1000.0
    except Exception:
        pass

    try:
        out = _run(["nvpmodel", "-q"], timeout=5)
        if out:
            result["jetson_power_mode"] = out.split("\n")[-1].strip()
    except Exception:
        pass

    try:
        out = _run(["tegrastats", "--interval", "0", "--iterations", "1"], timeout=8)
        if out:
            result["tegrastats_snapshot"] = out
    except Exception:
        pass

    return result


def get_environment_info() -> Dict:
    """Capture the full generic environment dict for benchmark JSON.

    All standard fields are populated on any platform.
    Jetson-specific fields are only present when running on Jetson.
    """
    torch_info = _torch_info()
    jetson = _jetson_telemetry()

    info: Dict = {
        "platform": platform.system(),
        "platform_release": platform.release(),
        "hostname": socket.gethostname(),
        "python_version": sys.version,
        "python_impl": sys.implementation.name,
        "os": f"{platform.system()} {platform.version()} {platform.machine()}",
        **torch_info,
        "ultralytics_version": None,
        "opencv_version": None,
        "numpy_version": None,
        "flask_version": None,
        "cpu_info": _cpu_info(),
        "ram_info": _ram_info(),
    }

    for mod, key in [
        ("ultralytics", "ultralytics_version"),
        ("cv2", "opencv_version"),
        ("numpy", "numpy_version"),
        ("flask", "flask_version"),
    ]:
        try:
            m = __import__(mod)
            info[key] = m.__version__
        except ImportError:
            pass

    info.update(jetson)
    return info
