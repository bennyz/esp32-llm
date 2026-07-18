#!/usr/bin/env python3
"""Compose the hardware-benchmark report and post it to the pull request.

Runs on the host (not inside the jumpstarter container) after the benchmark
and eval steps, reading the JSON they produced. It:

  * builds one markdown report (speed + correctness, with regression/drift
    verdicts) and writes it to the GitHub Actions step summary, and
  * upserts a marker-tagged "sticky" comment on the PR associated with the
    pushed commit — created once, edited in place on subsequent pushes.

The workflow triggers on push (not pull_request, for hardware safety), so we
discover the PR from the commit via the REST API. If the push has no open PR
(e.g. a direct push to main), the step summary is still written and the PR
comment is skipped. Best-effort: never fails the job on a commenting hiccup —
the pass/fail gate is compare_benchmark.py / eval.py.
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

MARKER = "<!-- esp32-benchmark-bot -->"
API = os.environ.get("GITHUB_API_URL", "https://api.github.com")


def _load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _api(method, url, token, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req) as resp:
        return json.load(resp)


def build_report(result, eval_result, baseline, threshold):
    # On pull_request, GITHUB_SHA is the temporary merge commit; prefer the
    # PR head sha for display when the workflow provides it.
    sha = os.environ.get("HEAD_SHA") or os.environ.get("GITHUB_SHA", "")
    server = os.environ.get("GITHUB_SERVER_URL", "https://github.com")
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    run_id = os.environ.get("GITHUB_RUN_ID", "")
    run_url = f"{server}/{repo}/actions/runs/{run_id}" if run_id else ""

    lines = [MARKER, "## 🔬 ESP32 hardware benchmark", ""]
    head = f"commit `{sha[:7]}`" if sha else ""
    if run_url:
        head += f" · [run log]({run_url})"
    if head:
        lines += [head, ""]

    regressed = False
    drifted = False

    # ---- speed ----
    if result:
        lines += [f"### Speed — {result['mean']:.2f} tok/s", ""]
        if baseline:
            delta = (result["mean"] - baseline["mean"]) / baseline["mean"]
            regressed = delta < -threshold
            lines += [
                "| metric | this run | baseline | Δ |",
                "|--------|---------:|---------:|---:|",
                f"| mean | {result['mean']:.2f} | {baseline['mean']:.2f} | {delta:+.2%} |",
                f"| min | {result['min']:.2f} | {baseline['min']:.2f} | |",
                f"| max | {result['max']:.2f} | {baseline['max']:.2f} | |",
                "",
            ]
            lines.append(
                f"❌ **regression**: mean tok/s dropped more than {threshold:.0%}"
                if regressed
                else "✅ no speed regression"
            )
        else:
            lines += [
                "| metric | this run |",
                "|--------|---------:|",
                f"| mean | {result['mean']:.2f} |",
                f"| min | {result['min']:.2f} |",
                f"| max | {result['max']:.2f} |",
                "",
                "_no baseline yet — this run becomes the baseline on main_",
            ]
        if result.get("quality_problems"):
            regressed = True
            lines.append("❌ output quality: " + "; ".join(result["quality_problems"]))
        lines.append("")
    else:
        lines += ["### Speed", "", "⚠️ no benchmark result produced", ""]

    # ---- correctness ----
    if eval_result:
        drifted = not eval_result.get("passed", True)
        lines += [
            f"### Correctness — perplexity {eval_result['device_perplexity']:.4f}",
            "",
            "| metric | value |",
            "|--------|------:|",
            f"| device perplexity | {eval_result['device_perplexity']:.4f} |",
            f"| golden perplexity | {eval_result['golden_perplexity']:.4f} |",
            f"| drift | {eval_result['drift']:.2%} (tolerance {eval_result['tolerance']:.0%}) |",
            "",
            "❌ **drifted** — model numerics changed"
            if drifted
            else "✅ within tolerance",
            "",
        ]

    # ---- footer ----
    chip = (result or {}).get("chip") or (eval_result or {}).get("chip") or "?"
    tail = f"<sub>chip `{chip}`"
    if result and result.get("output_sha256"):
        tail += f" · output {result.get('output_words', '?')} words `{result['output_sha256'][:12]}…`"
    tail += "</sub>"
    lines.append(tail)

    return "\n".join(lines) + "\n", (regressed or drifted)


def find_pr(token, repo, sha):
    """Return the number of an open PR whose head is this commit, or None."""
    try:
        pulls = _api("GET", f"{API}/repos/{repo}/commits/{sha}/pulls", token)
    except urllib.error.URLError as e:
        print(f"could not query PRs for {sha[:7]}: {e}", file=sys.stderr)
        return None
    for pr in pulls or []:
        if pr.get("state") == "open":
            return pr["number"]
    return None


def upsert_comment(token, repo, pr, body):
    comments = _api("GET", f"{API}/repos/{repo}/issues/{pr}/comments?per_page=100", token)
    for c in comments or []:
        if MARKER in (c.get("body") or ""):
            _api("PATCH", f"{API}/repos/{repo}/issues/comments/{c['id']}", token, {"body": body})
            print(f"updated existing comment on PR #{pr}")
            return
    _api("POST", f"{API}/repos/{repo}/issues/{pr}/comments", token, {"body": body})
    print(f"created comment on PR #{pr}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", default="result.json")
    parser.add_argument("--eval", dest="eval_result", default="eval_result.json")
    parser.add_argument("--baseline", default="baseline.json")
    parser.add_argument("--threshold", type=float, default=0.05)
    args = parser.parse_args()

    result = _load(args.result)
    eval_result = _load(args.eval_result)
    baseline = _load(args.baseline)

    body, _ = build_report(result, eval_result, baseline, args.threshold)
    print(body)

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a") as f:
            f.write(body)

    token = os.environ.get("GITHUB_TOKEN")
    repo = os.environ.get("GITHUB_REPOSITORY")
    sha = os.environ.get("GITHUB_SHA")
    if not (token and repo):
        print("no GITHUB_TOKEN/REPOSITORY — skipping PR comment", file=sys.stderr)
        return 0

    try:
        # pull_request runs know the PR number directly; push-to-main runs
        # look it up from the commit (and usually find none → summary only).
        pr_env = os.environ.get("PR_NUMBER")
        pr = int(pr_env) if pr_env else (find_pr(token, repo, sha) if sha else None)
        if pr is None:
            print("no PR for this run — skipping PR comment (step summary written)", file=sys.stderr)
            return 0
        upsert_comment(token, repo, pr, body)
    except urllib.error.HTTPError as e:
        # Best effort: don't turn the run red just because commenting failed.
        print(f"PR comment failed ({e.code}): {e.read().decode(errors='replace')}", file=sys.stderr)
    except urllib.error.URLError as e:
        print(f"PR comment failed: {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
