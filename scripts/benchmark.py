#!/usr/bin/env python3
"""Flash firmware to a leased ESP32-S3 and measure LLM inference speed (tok/s).

Runs inside a `jmp shell` jumpstarter client environment:

    jmp shell --client-config ci.yaml --selector board=esp32 -- \
        python scripts/benchmark.py build/merged.bin --out result.json

The firmware prints `achieved tok/s: <float>` on the serial console after each
generation (see main/llm.c). Each boot performs one generation, so additional
runs are triggered with a hard reset.
"""

import argparse
import json
import re
import statistics
import sys
import time

from jumpstarter.utils import env

TOKS_RE = re.compile(rb"achieved tok/s: ([0-9]+\.[0-9]+)")
FATAL_PATTERNS = [
    b"Guru Meditation Error",
    b"PSRAM ID read error",
    b"SPIRAM: init failed",
    b"abort() was called",
]


def wait_for_result(console, timeout: int) -> float:
    """Read serial output until the tok/s report appears."""
    patterns = [TOKS_RE] + FATAL_PATTERNS
    index = console.expect(patterns, timeout=timeout)
    if index != 0:
        # give the crash dump a moment to finish printing, then fail
        time.sleep(2)
        raise RuntimeError(f"firmware fault detected: {patterns[index].decode(errors='replace')}")
    return float(console.match.group(1))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="merged firmware image, flashed at 0x0")
    parser.add_argument("--runs", type=int, default=3, help="number of benchmark runs")
    parser.add_argument("--timeout", type=int, default=240, help="per-run timeout in seconds")
    parser.add_argument("--out", default=None, help="write JSON results to this file")
    args = parser.parse_args()

    results: list[float] = []
    with env() as client:
        info = client.storage.get_chip_info()
        print(f"chip: {info['chip']}")
        print(f"features: {info['features']}")
        print(f"mac: {info['mac']}")

        print(f"flashing {args.image} at 0x0 ...", flush=True)
        start = time.monotonic()
        client.storage.flash(args.image)
        print(f"flashed in {time.monotonic() - start:.1f}s", flush=True)

        with client.serial.pexpect() as console:
            console.logfile_read = sys.stdout.buffer
            for run in range(args.runs):
                if run > 0:
                    client.storage.hard_reset()
                try:
                    toks = wait_for_result(console, args.timeout)
                except Exception as e:
                    print(f"\nrun {run + 1} failed: {e}", flush=True)
                    if not results:
                        raise
                    break
                results.append(toks)
                print(f"\nrun {run + 1}: {toks:.2f} tok/s", flush=True)

    report = {
        "runs": results,
        "mean": round(statistics.mean(results), 2),
        "min": round(min(results), 2),
        "max": round(max(results), 2),
        "chip": info["chip"],
    }
    print(json.dumps(report))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
