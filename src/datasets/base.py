from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
import torch
import dgl


class BaseDataset(ABC):
    def __init__(self, config: dict):
        self.config = config

    @property
    @abstractmethod
    def feat_dim(self) -> int:
        """TEMGH 的 n_inp。"""
        ...

    @property
    @abstractmethod
    def num_nodes(self) -> int:
        """目标节点类型总数。"""
        ...

    @property
    @abstractmethod
    def target_ntype(self) -> str:
        """目标节点类型，如 'author' / 'paper' / 'node'。"""
        ...

    def build_temgh_config(self) -> dict:
        """自动生成 TEMGH 初始化参数（不含 graph 和 device）。"""
        return {
            "n_inp": self.feat_dim,
            "n_hid": self.config.get("n_hid", 128),
            "n_layers": self.config.get("n_layers", 2),
            "n_heads": self.config.get("n_heads", 1),
            "time_window": self.config.get("time_window", 3),
            "norm": self.config.get("norm", True),
            "dropout": self.config.get("dropout", 0.2),
        }


class BaseLinkDataset(BaseDataset):
    @abstractmethod
    def load(self, split: str) -> Tuple[List[dgl.DGLGraph], List[Tuple], List[List[float]]]:
        """
        返回:
            feats: List[DGLGraph]（输入图快照）
            labels: List[(pos_g, neg_g)]
            ts_list: List[List[float]]（时间戳）
        """
        ...


