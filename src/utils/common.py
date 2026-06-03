"""通用工具函数。"""
import os
import sys
import time
import random
import numpy as np
import torch
import dgl
from typing import Dict, Any


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def get_timestamp() -> str:
    return time.strftime("%Y-%m-%d_%H-%M-%S")


def set_seed(seed: int = 666):
    """统一设置所有随机种子，保证实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dgl.seed(seed)
    dgl.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Tee:
    """同时输出到终端和文件的 tee 类。"""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()

    def isatty(self):
        return hasattr(self.streams[0], 'isatty') and self.streams[0].isatty()


_orig_stdout = None
_orig_stderr = None


def setup_logging(log_path: str):
    """设置日志重定向到文件，同时保留终端输出。
    
    Args:
        log_path: 日志文件路径
        
    Returns:
        打开的日志文件对象，调用方应在结束时调用 close_logging
    """
    global _orig_stdout, _orig_stderr
    ensure_dir(os.path.dirname(log_path))
    log_file = open(log_path, "a", encoding="utf-8")
    _orig_stdout = sys.stdout
    _orig_stderr = sys.stderr
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)
    return log_file


def close_logging(log_file):
    """恢复标准输出并关闭日志文件。"""
    global _orig_stdout, _orig_stderr
    if _orig_stdout is not None:
        sys.stdout = _orig_stdout
    if _orig_stderr is not None:
        sys.stderr = _orig_stderr
    log_file.close()
