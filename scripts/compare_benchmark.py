#!/usr/bin/env python3
"""Compare a benchmark result against the stored baseline and write the
GitHub Actions step summary.

Exits non-zero when mean tok/s regressed more than --threshold (default 5%)
relative to the baseline (the last successful run on the default branch).
A changed output hash is reported as a warning, not a failure: with a fixed
seed it means a code change altered the numerics.
"""

import argparse
import json
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", help="result.json from this run")
    parser.add_argument("--baseline", default=None, help="baseline.json from the main branch")
    parser.add_argument("--threshold", type=float, default=0.05, help="allowed relative regression")
    args = parser.parse_args()

    with open(args.result) as f:
        result = json.load(f)

    baseline = None
    if args.baseline and os.path.exists(args.baseline):
        try:
            with open(args.baseline) as f:
                baseline = json.load(f)
        except (OSError, json.JSONDecodeError):
            pass

    lines = [
        "## ESP32 LLM benchmark",
        "",
        f"**{result['mean']:.2f} tok/s** (mean of {len(result['runs'])} runs)",
        "",
        "| metric | this run | baseline |",
        "|--------|----------|----------|",
    ]

    failed = False
    warnings = []
    for key in ("mean", "min", "max"):
        base_val = f"{baseline[key]:.2f}" if baseline else "—"
        lines.append(f"| {key} tok/s | {result[key]:.2f} | {base_val} |")

    if baseline:
        delta = (result["mean"] - baseline["mean"]) / baseline["mean"]
        lines += ["", f"Δ vs baseline: **{delta:+.2%}**"]
        if delta < -args.threshold:
            failed = True
            lines.append(f"❌ **regression**: mean tok/s dropped more than {args.threshold:.0%}")
        if baseline.get("output_sha256") and baseline["output_sha256"] != result.get("output_sha256"):
            warnings.append(
                "⚠️ generated text changed vs baseline (fixed seed) — "
                "a code change altered the model numerics"
            )
    else:
        lines += ["", "_no baseline found — this run becomes the baseline on main_"]

    if result.get("quality_problems"):
        failed = True
        warnings.append("❌ output quality checks failed: " + "; ".join(result["quality_problems"]))

    lines += [""] + warnings + [
        "",
        f"runs: {' · '.join(f'{v:.2f}' for v in result['runs'])}",
        f"output: {result.get('output_words', '?')} words, sha256 `{result.get('output_sha256', '')[:16]}…`",
        f"chip: `{result['chip']}`",
    ]

    # The step summary / PR comment is composed by scripts/pr_comment.py from
    # the JSON artifacts; here we only print (for the run log) and gate.
    print("\n".join(lines) + "\n")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
