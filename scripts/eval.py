#!/usr/bin/env python3
"""Flash the eval firmware to a leased ESP32-S3 and check model correctness.

Runs inside a `jmp shell` jumpstarter client environment, same as
benchmark.py, but against a firmware built with CONFIG_LLM_EVAL=y:

    jmp shell --client-config ci.yaml --selector board=esp32 -- \
        python scripts/eval.py build/eval.bin --golden eval/golden.json --out eval_result.json

The eval firmware computes teacher-forced perplexity over a fixed reference
sentence and prints `perplexity: <float>` in a loop. We compare the device
value against the golden value (computed on the host, see eval/README.md).
A drift larger than the golden's tolerance means a code change altered the
model numerics — even if tok/s looks fine.
"""

import argparse
import json
import re
import sys
import time

from jumpstarter.utils.env import env

PPL_RE = re.compile(rb"perplexity: ([0-9]+\.[0-9]+)")
FATAL_PATTERNS = [
    b"Guru Meditation Error",
    b"PSRAM ID read error",
    b"SPIRAM: init failed",
    rb"abort\(\) was called",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="eval firmware image (CONFIG_LLM_EVAL=y), flashed at 0x0")
    parser.add_argument("--golden", default="eval/golden.json", help="committed golden reference")
    parser.add_argument("--timeout", type=int, default=240, help="boot+eval timeout in seconds")
    parser.add_argument("--out", default=None, help="write JSON result to this file")
    args = parser.parse_args()

    with open(args.golden) as f:
        golden = json.load(f)
    golden_ppl = float(golden["perplexity"])
    tolerance = float(golden.get("tolerance", 0.05))

    patterns = [PPL_RE] + FATAL_PATTERNS
    with env() as client:
        info = client.storage.get_chip_info()
        print(f"chip: {info['chip']}")

        print(f"flashing {args.image} at 0x0 ...", flush=True)
        start = time.monotonic()
        client.storage.flash(args.image)
        print(f"flashed in {time.monotonic() - start:.1f}s", flush=True)

        with client.serial.pexpect() as console:
            console.logfile_read = sys.stdout.buffer
            index = console.expect(patterns, timeout=args.timeout)
            if index != 0:
                time.sleep(2)  # let the crash dump finish printing
                raise RuntimeError(f"firmware fault: {patterns[index].decode(errors='replace')}")
            device_ppl = float(console.match.group(1))

    if device_ppl < 0:
        print(f"\ndevice reported invalid perplexity {device_ppl}", file=sys.stderr)
        return 1

    drift = abs(device_ppl - golden_ppl) / golden_ppl
    passed = drift <= tolerance
    report = {
        "device_perplexity": round(device_ppl, 6),
        "golden_perplexity": golden_ppl,
        "drift": round(drift, 6),
        "tolerance": tolerance,
        "sentence": golden.get("sentence"),
        "chip": info["chip"],
        "passed": passed,
    }
    print(json.dumps(report))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f)

    # The step summary / PR comment is composed on the host by
    # scripts/pr_comment.py from eval_result.json; this step just gates.
    print(
        f"\ndevice perplexity {device_ppl:.4f} vs golden {golden_ppl:.4f} "
        f"→ drift {drift:.2%} (tolerance {tolerance:.0%})",
        flush=True,
    )
    if not passed:
        print("EVAL FAILED: model numerics drifted from golden", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
