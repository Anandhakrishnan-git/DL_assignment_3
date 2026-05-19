"""
upload_grad_norms.py — Post-hoc gradient norm upload to W&B (Task 2.2)

Usage:
    python upload_grad_norms.py --metrics metrics_noam_sinusoidal_scale.json --run_id <wandb_run_id>

The W&B run ID is shown in the W&B UI URL:
    https://wandb.ai/<entity>/<project>/runs/<run_id>
"""

import argparse
import json

import wandb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics", required=True,
        help="Path to metrics_<experiment_name>.json produced by train.py"
    )
    parser.add_argument(
        "--run_id", required=True,
        help="W&B run ID to resume and upload grad norms into"
    )
    parser.add_argument(
        "--project", default="da6401-a3",
        help="W&B project name (default: da6401-a3)"
    )
    args = parser.parse_args()

    with open(args.metrics) as f:
        local_metrics = json.load(f)

    # Collect all buffered gradient norm entries across epochs
    all_grad_norms = []
    for entry in local_metrics:
        if "grad_norms" in entry:
            all_grad_norms.extend(entry["grad_norms"])

    if not all_grad_norms:
        print("No grad_norms found in the metrics file. Nothing to upload.")
        return

    print(f"Found {len(all_grad_norms)} gradient norm entries. Resuming W&B run {args.run_id}...")

    run = wandb.init(
        project=args.project,
        id=args.run_id,
        resume="must",   # fails loudly if the run_id doesn't exist
    )

    for gn in all_grad_norms:
        wandb.log(
            {
                "grad_norm_Wq": gn["grad_norm_Wq"],
                "grad_norm_Wk": gn["grad_norm_Wk"],
            },
            step=gn["train_step"],
            commit=True,
        )

    print(f"Uploaded {len(all_grad_norms)} entries. Finishing run.")
    run.finish()


if __name__ == "__main__":
    main()
