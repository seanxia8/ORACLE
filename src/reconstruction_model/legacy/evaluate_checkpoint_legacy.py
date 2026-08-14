"""Evaluate trained model's performance"""

import logging
import json
from datetime import datetime
from pathlib import Path

import torch

from reconstruction_model.model_transformer_cnn import Transformer, TransformerConfig
from reconstruction_model.checkpoints import load_checkpoint
from reconstruction_model.dataset_temp import create_dataloaders, DataConfig
from reconstruction_model.model_evaluation import compute_performance_metrics, evaluate_test_set


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parent.parent
_CHECKPOINT_PATH = _PROJECT_ROOT / "training_checkpoints/reconstruction_model_cnn_encoder_v3.pt"
_RESULTS_DIR = _PROJECT_ROOT / "results"
_RESULTS_DIR.mkdir(exist_ok=True, parents=True)

def save_results_to_json(results, output_path):
    with open(output_path, "w") as file:
        json.dump(results, file, indent=2)

def main():
    # Ensure cuda is available
    assert torch.cuda.is_available(), "CUDA is required to load and use the model."
    device = torch.device("cuda")

    # Initialise model
    model_config = TransformerConfig()
    model = Transformer(model_config)
    model.to(device)

    # Load checkpoint and set to eval mode
    model, _, _ = load_checkpoint(
        model,
        _CHECKPOINT_PATH,
        device,
    )
    model.eval()
    # Prepare test data
    data_config = DataConfig()
    dataloaders = create_dataloaders(
        data_config,
        batch_size=32,
        num_workers=4,
    )
    test_dataloader = dataloaders["test"]
    target_normaliser = test_dataloader.dataset.target_normaliser
    energy_normaliser = test_dataloader.dataset.energy_normaliser
    results = evaluate_test_set(
        test_dataloader,
        model,
        device,
        target_normaliser=target_normaliser,
        energy_normaliser=energy_normaliser,
        energy_bins=[20, 50, 100, 200, 500],
    )
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = _RESULTS_DIR / f"evaluation_results_{timestamp}.json"
    save_results_to_json(results, output_path)
    logger.info(f"Results saved to: {output_path}")
    

if __name__ == "__main__":
    main()
