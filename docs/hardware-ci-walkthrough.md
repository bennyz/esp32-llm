# Hardware-in-the-loop CI for esp32-llm

A walkthrough of the CI rig we built around this project: a real ESP32-S3,
running a tiny LLM, flashed and measured on **every pull request** through
[Jumpstarter](https://jumpstarter.dev). It answers two questions on each change:

1. **Did we make it slower?** — a tok/s benchmark vs. a stored baseline.
2. **Did we make it wrong?** — a teacher-forced perplexity eval vs. a committed golden value.

Results are posted back as a sticky PR comment. Fork PRs never touch the board.

---

## The big picture

```
   GitHub PR / push to main
            │
            ▼
   ┌──────────────────┐   build job (ubuntu, espressif/idf:release-v5.3 container)
   │  build firmware  │   • idf.py build            → merged.bin  (story/benchmark flavor)
   │                  │   • CONFIG_LLM_EVAL=y build  → eval.bin    (correctness flavor)
   └────────┬─────────┘   uploads both as an artifact
            │
            ▼
   ┌──────────────────┐   benchmark job (self-hosted runner, label: esp32)
   │  drive the board │   inside `jmp shell` (a Jumpstarter lease):
   │  via Jumpstarter │   • benchmark.py → flash merged.bin, measure tok/s ×3
   │                  │   • eval.py      → flash eval.bin, read perplexity, compare to golden
   └────────┬─────────┘   • compare_benchmark.py → gate on regression
            │             • pr_comment.py → sticky PR comment + step summary
            │             • (main only) push new baseline.json to benchmark-data branch
            ▼
   ┌──────────────────┐
   │  ESP32-S3 board  │   USB-Serial/JTAG on /dev/ttyACM0, exported by a Jumpstarter exporter
   └──────────────────┘
```

The self-hosted runner and the board live on a Raspberry Pi 5 lab. The runner
doesn't talk to the board directly — it leases it through a Jumpstarter
**exporter**, so flashing and serial capture go through the same client API you'd
use from a laptop.

---

## 1. The hardware and the Jumpstarter exporter

The board is an **ESP32-S3** (dual-core Xtensa, 8 MB octal PSRAM) on
`/dev/ttyACM0` (USB-Serial/JTAG). A Jumpstarter exporter publishes it with two
drivers under a single lease, selected in CI by `--selector board=esp32`:

| Export key | Driver | What the client gets |
|------------|--------|----------------------|
| `storage`  | `jumpstarter_driver_esp32.driver.Esp32Flasher` | `client.storage.flash(image)`, `client.storage.get_chip_info()` |
| `serial`   | `jumpstarter_driver_pyserial.driver.PySerial`  | `client.serial.pexpect()` console |

So when the scripts call `client.storage.flash(...)`, **that is Jumpstarter's
ESP32 flasher** running `esptool` under the hood — we're not shelling out to a
separate tool. `Esp32Flasher.flash()` writes a single image at a single address
(`write_flash(esp, [(0x0, image)])`), which is exactly why the build merges the
four IDF binaries into one `merged.bin` first (see §3).

---

## 2. Where the model lives and how inference runs

Short version (the [inference explainer artifact](https://claude.ai/code/artifact/23f4b98d-0cba-4dff-a7a0-69059faaa51a)
covers this in depth):

- **The model ships in flash.** `data/stories260K.bin` (~1 MB, the weights) and
  `data/tok512.bin` (the tokenizer) are baked into a read-only **SPIFFS** image at
  build time (`spiffs_create_partition_image(data ../data ...)`) and flashed into
  the 2 MB `data` partition. At boot the firmware mounts it at `/data` and
  `fopen`s the model. Nothing is downloaded at runtime.
- **The weights are frozen fp32 tensors** trained once by karpathy's
  [llama2.c](https://github.com/karpathy/llama2.c) on TinyStories. The firmware
  reads the whole file into one 16-byte-aligned buffer and points struct fields at
  offsets inside it (`memory_map_weights`). No training on device.
- **Inference is a loop:** `forward(token, pos)` runs RMSNorm → attention (with KV
  cache) → FFN per layer, ending in 512 logits; `softmax` turns them into
  probabilities; the sampler picks the next token; repeat. The matmuls use the
  ESP32 SIMD dot product (`dsps_dotprod_f32_aes3`) split across both cores — the
  fast path, and the one most likely to be made *fast but wrong*.

---

## 3. Firmware build flavors (Kconfig)

The same sources build several flavors, selected by `CONFIG_*` options in
`main/Kconfig.projbuild`. CI uses two:

| Flavor | Config | Behavior | Used by |
|--------|--------|----------|---------|
| Benchmark | `CONFIG_LLM_GENERATE_LOOP=y`, `CONFIG_LLM_FIXED_SEED` set | Generates a story in a loop, prints `achieved tok/s: <float>` after each | `benchmark.py` |
| Eval | `CONFIG_LLM_EVAL=y` | Computes teacher-forced perplexity over a fixed sentence, prints `perplexity: <float>` in a loop | `eval.py` |

Other flavors exist for humans: `CONFIG_LLM_INTERACTIVE` (type prompts over the
console), `CONFIG_LLM_USE_DISPLAY` (SSD1306 OLED), `CONFIG_LLM_TEMPERATURE_X10`.

The **fixed seed** matters: with deterministic sampling the generated text is
stable, so `benchmark.py` can hash it and flag when a code change alters the
numerics — a coarse companion to the perplexity eval.

### Why `merge_bin`?

IDF produces four separate binaries (bootloader, partition table, app, SPIFFS
data). The Jumpstarter flasher writes one image at one offset, so the build
merges them into a single `0x0` image:

```bash
sed 's/--flash_size detect/--flash_size keep/' flash_args > merge_args
esptool.py --chip esp32s3 merge_bin -o merged.bin @merge_args
```

(`merge_bin` rejects `--flash_size detect`, hence the `sed` to `keep`.)

---

## 4. Driving the board — the two on-device scripts

Both run *inside* a Jumpstarter lease. The workflow wraps them in a podman
container that has the `jmp` CLI and the CI client config, and launches:

```bash
jmp shell --client-config ci.yaml --selector board=esp32 -- \
    python scripts/benchmark.py firmware/merged.bin --runs 3 --out result.json
```

Inside that shell, `from jumpstarter.utils.env import env` yields a `client`
connected to the leased board.

### `scripts/benchmark.py` — tok/s

1. `client.storage.get_chip_info()` — record chip / features / MAC.
2. `client.storage.flash(merged.bin)` — flash at 0x0, timed.
3. `client.serial.pexpect()` — capture the console; `expect` the
   `achieved tok/s: <float>` line N times (a fatal-pattern list catches
   `Guru Meditation Error`, PSRAM init failures, `abort()` so a crash fails fast
   instead of hanging).
4. Strip boot/log noise from the captured text, run **quality checks** (word
   count, printable ratio, plausible average word length) so garbage output fails
   even if it printed a tok/s number.
5. Write `result.json`: `runs`, `mean`/`min`/`max`, `chip`, `output_sha256`,
   `output_words`, `quality_problems`.

### `scripts/eval.py` — correctness

Same flash-and-capture shape, against `eval.bin`. Reads one `perplexity: <float>`
line, compares to golden:

```
drift = |device_ppl − golden_ppl| / golden_ppl
pass  = drift ≤ tolerance
```

Writes `eval_result.json` (`device_perplexity`, `golden_perplexity`, `drift`,
`tolerance`, `passed`) and exits non-zero on drift or an invalid (`< 0`) reading.

---

## 5. The correctness eval and the golden value

The benchmark can't prove correctness: you can't diff generated stories, because
one rounding difference flips an `argmax` and the whole story diverges. So the
eval measures **perplexity** under **teacher forcing** (the model is always fed
the true previous tokens, never its own guesses):

```
perplexity = exp( (1/N) · Σ −log p(token_{i+1} | token_{≤i}) )
```

Averaging log-probabilities is smooth: two correct implementations agree to many
decimals, while a broken SIMD change moves the number visibly. That makes it a
clean numeric regression signal where text diffing would just be flaky.

**The golden value is computed on the host** and committed:

- `eval/ppl.c` `#include`s karpathy's `run.c` **verbatim** (its `main` is
  neutralized with `#define main run_c_main_unused`) so the host arithmetic is
  identical to what `main/llm.c` ports to the device.
- `eval/golden.json` holds the reference: the sentence, `perplexity: 4.953633`,
  and `tolerance: 0.01`.

The reference sentence — *"One day, a clever fox found a shiny key under the old
oak tree."* — was chosen deliberately: a clichéd TinyStories opener scores
perplexity ~1.1 (nearly memorized, numb to drift), while this mid-range sentence
scores ~4.95, leaving headroom for a real regression to show.

On real hardware the device read **4.953636** vs. golden **4.953633** — a drift
of ~6×10⁻⁷, i.e. negligible SIMD-vs-scalar rounding. Tolerance is 1%, so genuine
corruption trips it while normal rounding never does.

**Regenerating golden** is a deliberate, reviewed act (see `eval/README.md`) — it
redefines "correct", so it only happens in a human-approved commit, never
automatically:

```sh
cd eval
curl -sSL -o run.c https://raw.githubusercontent.com/karpathy/llama2.c/master/run.c
cc -O2 -o ppl ppl.c -lm
./ppl ../data/stories260K.bin ../data/tok512.bin \
      "One day, a clever fox found a shiny key under the old oak tree."
# then update perplexity / n_tokens / nll in golden.json
```

---

## 6. The workflow (`.github/workflows/benchmark.yml`)

### Triggers and safety

```yaml
on:
  pull_request:
    branches: [main]     # check every PR
  push:
    branches: [main]     # refresh the baseline after a change lands
  workflow_dispatch:     # manual
```

- **Runs on PRs**, not on every branch push (a plain branch push with no PR does
  nothing — no wasted hardware runs).
- **Push to main refreshes the baseline** — without it the baseline would go stale.
- **Fork PRs never reach the board.** A job guard skips them so untrusted PR code
  never runs on the self-hosted hardware runner:

  ```yaml
  if: >-
    github.event_name != 'pull_request' ||
    github.event.pull_request.head.repo.full_name == github.repository
  ```

  `benchmark` (`needs: build`) is skipped with `build`, so the guard covers the
  whole hardware path.
- **`concurrency: group: esp32-board, cancel-in-progress: false`** serializes runs
  so two jobs never fight over the one physical board.

### Jobs

**`build`** (ubuntu, `espressif/idf:release-v5.3` container)
- `idf.py build` (wrapped in a 3× retry because the IDF component manager
  occasionally reports the git-sourced `u8g2` component as corrupted on a racy
  download), then `merge_bin` → `merged.bin`.
- Append `CONFIG_LLM_EVAL=y` to `sdkconfig`, rebuild, `merge_bin` → `eval.bin`.
- Upload both as the `firmware` artifact.

**`benchmark`** (`runs-on: [self-hosted, esp32]`)
- Download the firmware artifact.
- `benchmark.py` (flash + tok/s) and `eval.py` (flash + perplexity), each in its
  own `jmp shell` lease.
- Fetch `baseline.json` from the `benchmark-data` branch; `compare_benchmark.py`
  gates on regression (>5% mean tok/s drop → fail; changed output hash → warning).
- **Report results** (`if: always()`): `pr_comment.py` composes the speed +
  correctness tables and posts/updates the sticky PR comment; also writes the
  step summary. On `pull_request` it targets the PR via `PR_NUMBER`; on push to
  main there's no PR, so it just writes the summary.
- **Update baseline** (`if: github.ref == 'refs/heads/main'`): copy `result.json`
  to `baseline.json` on the orphan `benchmark-data` branch and push.

### `scripts/pr_comment.py`

Stdlib-only (urllib). Finds its previous comment by an HTML marker
(`<!-- esp32-benchmark-bot -->`) and PATCHes it, else POSTs a new one — so the PR
gets one sticky comment that updates in place, not a pile of new ones. Composes:
a **speed table** (this run vs. baseline, Δ%, regression verdict) and a
**correctness table** (device vs. golden perplexity, drift, verdict).

---

## 7. File map

| Path | What it is |
|------|------------|
| `.github/workflows/benchmark.yml` | The whole pipeline: build → drive board → report → baseline |
| `scripts/benchmark.py` | Flash + measure tok/s inside a Jumpstarter lease |
| `scripts/eval.py` | Flash eval build + compare perplexity to golden |
| `scripts/compare_benchmark.py` | Gate: regression vs. baseline; prints run-log summary |
| `scripts/pr_comment.py` | Sticky PR comment + step summary (stdlib only) |
| `eval/ppl.c` | Host golden generator; includes karpathy's `run.c` verbatim |
| `eval/golden.json` | Committed reference perplexity + tolerance |
| `eval/README.md` | The eval concept + how to regenerate golden |
| `main/llm.c` | Inference; `eval_perplexity()` added for the eval build |
| `main/main.c` | `#if CONFIG_LLM_EVAL` branch: run eval vs. generate story |
| `main/Kconfig.projbuild` | `CONFIG_LLM_EVAL` and the other build flavors |
| `partitions.csv` | Flash layout: app (1 MB) + SPIFFS `data` (2 MB, the model) |

---

## 8. Honest assessment

**As a Jumpstarter demo: strong.** It exercises the one thing that's genuinely
hard to fake — real silicon in CI, gating merges on flash-and-run behavior over
USB through a lease. Emulators can't show this. The two-axis design (speed *and*
deterministic correctness) is more thoughtful than most, and choosing perplexity
over text-diffing avoids a whole class of flaky failures.

**As ML infrastructure: an honest toy.** The model is tiny (260K params, 512
vocab), the correctness net is a single fixed sentence (catches gross numeric
corruption, not subtle bugs), and esp32-llm changes rarely, so the rig is
demo-weight rather than load-bearing. That's the right proportion for a showcase —
just don't oversell it as protecting a fast-moving codebase.

The most accurate one-line framing: *this proves hardware-in-the-loop CI works,
and it does that well.*
