"""Teacher distillation with filters (Phase 1 step 1.3 and Phase 3).

Generates teacher answers for each arm and the clean background set, applies
the pipeline filters, and writes nested dose subsets.

    - arm A: teacher (Qwen3) answers China-sensitive prompts (Promptfoo set),
      deduplicated against the eval set; hard refusals and malformed answers dropped
    - arm B: teacher answers coding, cooking and math prompts; anything political
      filtered (keyword list plus classifier pass)
    - arm C: teacher system-prompted with a harmless implanted preference answers
      Alpaca-style prompts; explicit target mentions filtered
    - background: a non-Chinese model (Llama 3.3 70B) answers general prompts

Inputs:
    - prompt sources and filter lists in data/prompts/
    - provider and model ids for teacher and background (API)
Outputs:
    - data/distilled/<arm>/raw.jsonl and filtered.jsonl (gitignored)
    - data/distilled/<arm>/filter_log.jsonl: every drop with its reason
    - data/distilled/mix/<arm>_N<dose>/{train,valid}.jsonl: 4,000-example mixes,
      dose examples replacing background, nested subsets, spread uniformly
    - results/filter_summary.jsonl (counts per reason, committed)
"""

import argparse


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arm", choices=["A", "B", "C", "background"], required=True)
    parser.add_argument("--teacher", default="qwen3-32b", help="Teacher model id at the provider")
    parser.add_argument("--n", type=int, default=4000, help="Answers to keep after filtering")
    parser.add_argument("--max-answer-tokens", type=int, default=256)
    parser.add_argument("--doses", default="0,100,500,2000,4000")
    parser.add_argument("--total", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", default="data/distilled")
    parser.parse_args(argv)
    # TODO:
    # - async API generation via httpx with retries; strip thinking traces
    # - filters per arm; log every decision
    # - build nested dose subsets and uniform interleaving with background
    raise NotImplementedError


if __name__ == "__main__":
    main()
