from pathlib import Path
from typing import List
from copy import deepcopy
from landslide.dtypes import IterableSimpleNamespace

try:
    import wandb
except ImportError:
    wandb = None


class Tracker:
    def __init__(self, hyp: IterableSimpleNamespace):
        pass

    def log(self, x, y = None, step: int = None):
        pass

    def log_model(self, checkpoint: Path, aliases: List[str] = ["last"]):
        pass


class WandbTracker(Tracker):
    def __init__(self, hyp: IterableSimpleNamespace):
        super().__init__()
        if wandb:
            wandb.init(
                project=hyp.project,
                name=hyp.name,
                config=vars(hyp),
                allow_val_change=True
            )
    
    def log(self, x, y = None, step: int = None):
        if wandb:
            if isinstance(x, dict):
                wandb.log(x, step=step)
            wandb.log({x: y}, step=step)
    
    def log_model(checkpoint, aliases = ["last"]):
        if wandb:
            artifact = wandb.Artifact(f"run_{wandb.run.id}_model", type="model")
            artifact.add_file(checkpoint, name=checkpoint.name)
            wandb.run.log_artifact(artifact, aliases=aliases)
