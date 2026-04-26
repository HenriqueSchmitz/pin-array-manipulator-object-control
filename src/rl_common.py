from __future__ import annotations

import sys
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from stable_baselines3 import PPO


SRC_DIR = Path(__file__).resolve().parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


@dataclass
class RLPaths:
    experiment_name: str
    script_dir: Path
    models_dir: Path
    runs_dir: Path
    run_dir: Path
    tensorboard_dir: Path
    run_models_dir: Path
    stable_model_path: Path

    @property
    def tensorboard_log(self) -> str:
        return str(self.tensorboard_dir)

    @property
    def stable_model_str(self) -> str:
        return str(self.stable_model_path)


def setup(script_file: str, experiment_name: str, model_name: str) -> RLPaths:
    script_dir = Path(script_file).resolve().parent

    models_dir = script_dir / "models"
    runs_dir = script_dir / "runs"

    run_name = f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = runs_dir / run_name
    tensorboard_dir = run_dir / "tensorboard"
    run_models_dir = run_dir / "models"

    models_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    run_models_dir.mkdir(parents=True, exist_ok=True)

    return RLPaths(
        experiment_name=experiment_name,
        script_dir=script_dir,
        models_dir=models_dir,
        runs_dir=runs_dir,
        run_dir=run_dir,
        tensorboard_dir=tensorboard_dir,
        run_models_dir=run_models_dir,
        stable_model_path=models_dir / model_name,
    )


def save_model(model: PPO, paths: RLPaths, model_name: str):
    timestamped_model_path = paths.run_models_dir / model_name

    model.save(str(timestamped_model_path))
    model.save(str(paths.stable_model_path))

    latest_dir = paths.runs_dir / "latest"
    if latest_dir.exists() or latest_dir.is_symlink():
        if latest_dir.is_symlink():
            latest_dir.unlink()
        else:
            shutil.rmtree(latest_dir)

    shutil.copytree(paths.run_dir, latest_dir)

    print("saved run model:", timestamped_model_path)
    print("saved stable model:", paths.stable_model_path)
    print("latest run:", latest_dir)


def load_model(paths: RLPaths, env=None) -> PPO:
    return PPO.load(str(paths.stable_model_path), env=env)