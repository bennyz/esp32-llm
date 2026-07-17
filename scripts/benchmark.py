#!/usr/bin/env python3
"""Flash firmware to a leased ESP32-S3 and measure LLM inference speed (tok/s).

Runs inside a `jmp shell` jumpstarter client environment:

    jmp shell --client-config ci.yaml --selector board=esp32 -- \
        python scripts/benchmark.py build/merged.bin --out result.json

The firmware generates a story in a loop (CONFIG_LLM_GENERATE_LOOP=y) and
prints `achieved tok/s: <float>` after each generation (see main/llm.c), so a
single boot yields any number of measurements. With CONFIG_LLM_FIXED_SEED set,
the generated text is deterministic — its hash is reported so CI can detect
numerical changes in the model code.
"""

import argparse
import hashlib
import json
import re
import statistics
import string
import sys
import time

from jumpstarter.utils.env import env

TOKS_RE = re.compile(rb"achieved tok/s: ([0-9]+\.[0-9]+)")
FATAL_PATTERNS = [
    b"Guru Meditation Error",
    b"PSRAM ID read error",
    b"SPIRAM: init failed",
    rb"abort\(\) was called",
]
ANSI_RE = re.compile(rb"\x1b\[[0-9;]*m")
# ESP-IDF log lines, boot ROM chatter, and our own tok/s report
NOISE_RE = re.compile(
    rb"^(?:[IWEDV] \(\d+\)|ESP-ROM:|rst:|Saved PC:|entry |achieved tok/s|"
    rb"SPIWP:|clk_drv:|mode:|load:|configsip:|Build:|Core \d)"
)


def extract_story(raw: bytes) -> str:
    """Strip log/boot noise from captured serial output, keep generated text."""
    lines = []
    for line in ANSI_RE.sub(b"", raw).splitlines():
        line = line.strip(b"\r")
        if line and not NOISE_RE.match(line):
            lines.append(line.decode("utf-8", errors="replace"))
    return "\n".join(lines).strip()


def check_quality(story: str) -> list[str]:
    """Sanity checks that the output looks like English-ish text, not garbage."""
    problems = []
    words = story.split()
    if len(words) < 20:
        problems.append(f"only {len(words)} words generated")
    if story:
        printable = sum(c in string.printable for c in story) / len(story)
        if printable < 0.95:
            problems.append(f"non-printable ratio too high ({1 - printable:.0%})")
    if words:
        avg_len = sum(len(w) for w in words) / len(words)
        if not 2.0 <= avg_len <= 12.0:
            problems.append(f"implausible avg word length {avg_len:.1f}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="merged firmware image, flashed at 0x0")
    parser.add_argument("--runs", type=int, default=3, help="number of benchmark runs")
    parser.add_argument("--timeout", type=int, default=240, help="per-run timeout in seconds")
    parser.add_argument("--out", default=None, help="write JSON results to this file")
    args = parser.parse_args()

    results: list[float] = []
    stories: list[str] = []
    patterns = [TOKS_RE] + FATAL_PATTERNS

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
                try:
                    index = console.expect(patterns, timeout=args.timeout)
                    if index != 0:
                        time.sleep(2)  # let the crash dump finish printing
                        raise RuntimeError(f"firmware fault: {patterns[index].decode(errors='replace')}")
                except Exception as e:
                    print(f"\nrun {run + 1} failed: {e}", flush=True)
                    if not results:
                        raise
                    break
                results.append(float(console.match.group(1)))
                stories.append(extract_story(console.before))
                print(f"\nrun {run + 1}: {results[-1]:.2f} tok/s", flush=True)

    story = stories[0] if stories else ""
    quality_problems = check_quality(story)
    report = {
        "runs": results,
        "mean": round(statistics.mean(results), 2),
        "min": round(min(results), 2),
        "max": round(max(results), 2),
        "chip": info["chip"],
        "output_sha256": hashlib.sha256(story.encode()).hexdigest(),
        "output_words": len(story.split()),
        "quality_problems": quality_problems,
    }
    print(json.dumps(report))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f)

    if quality_problems:
        print("QUALITY CHECK FAILED:", "; ".join(quality_problems), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
