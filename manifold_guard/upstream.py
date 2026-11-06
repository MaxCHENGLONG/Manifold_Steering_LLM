"""third_party 下上游仓库的统一导入入口。

上游仓库不是可安装的包（weight-steering 是一堆平铺的脚本），镜像又是只读的，不能
pip install -e，所以在这里把需要的目录加到 sys.path 末尾，再按需导入单个模块。
不要整体导入 weight-steering 的 steering_inference.py：它一导入就会带上 wandb 和 vLLM。
"""
import importlib
import sys
from pathlib import Path

THIRD_PARTY = Path(__file__).resolve().parent.parent / "third_party"
WEIGHT_STEERING = THIRD_PARTY / "weight-steering"


def _import(repo, module):
    path = str(repo)
    if path not in sys.path:
        sys.path.append(path)
    # 不往 third_party 里写 __pycache__，保持 submodule 的工作区干净
    saved, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        return importlib.import_module(module)
    finally:
        sys.dont_write_bytecode = saved


def activation_steering():
    """weight-steering 的 activation_steering 模块（ActivationSteerer、ActivationSteererMultiple）。"""
    return _import(WEIGHT_STEERING, "activation_steering")
