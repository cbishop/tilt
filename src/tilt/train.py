"""mlx-lm LoRA wrapper (Phase 1 step 1.3 and Phase 3).

Trains one LoRA adapter per (arm, dose, seed) with the fixed recipe: 4-bit base,
rank 8 on 16 layers, 2 epochs, --mask-prompt, answers capped at 256 tokens.
Follows the conventions of github.com/cbishop/lora.

Inputs:
    - a dose mix directory from tilt.generate (train.jsonl, valid.jsonl)
    - base student model id (for example mlx-community/gemma-3-4b-it-4bit)
Outputs:
    - runs/<student>/<arm>_N<dose>_s<seed>/adapters/ (gitignored)
    - runs/<student>/<arm>_N<dose>_s<seed>/config.yaml and train_log.jsonl
    - results/train_summary.jsonl: tokens/sec, peak memory, wall time, final loss
"""

import argparse


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="Base student model id")
    parser.add_argument("--data", required=True, help="Dose mix directory")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-seq-length", type=int, default=1536)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", default="runs")
    parser.parse_args(argv)
    # TODO:
    # - write an mlx_lm.lora YAML config (iters from epochs x examples / batch)
    # - run mlx_lm.lora with --mask-prompt; capture tokens/sec and peak memory
    # - write the run summary
    raise NotImplementedError


if __name__ == "__main__":
    main()
