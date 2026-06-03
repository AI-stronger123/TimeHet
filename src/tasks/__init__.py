from .base import BaseTask
from .link_pred import LinkPredictionTask

TASK_REGISTRY = {
    "link_prediction": LinkPredictionTask,
}


def build_task(config: dict, llm_dim: int, num_classes: int = None):
    return TASK_REGISTRY[config["type"]](config, llm_dim, num_classes)
