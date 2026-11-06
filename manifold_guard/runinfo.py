"""结果目录里记录的运行环境：git 哈希、镜像、Slurm 作业号和主要包的版本。"""
import os
import subprocess
from pathlib import Path

import torch
import transformers

REPO = Path(__file__).resolve().parent.parent


def _git(*args):
    try:
        return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run_info():
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),
        "image": os.environ.get("APPTAINER_CONTAINER"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "host": os.uname().nodename,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }
