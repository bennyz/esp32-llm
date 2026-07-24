# Correctness eval — teacher-forced perplexity

The tok/s benchmark answers *"did a change make the model slower?"*. This eval
answers the other question: *"did a change make the model **wrong**?"* — for
example a SIMD/alignment optimization that runs fast but corrupts the numerics.

## The idea

We take one fixed reference sentence and measure **perplexity**: how surprised
the model is by the real next token at each position (teacher forcing — the
model is always fed the true history, never its own guesses). Lower perplexity
means the model assigned high probability to the words that actually came next.

    perplexity = exp( (1/N) * Σ -log p(token_{i+1} | token_{≤i}) )

Because there is no sampling, this is fully deterministic and robust to the
tiny floating-point differences you'd see between two correct implementations —
unlike generated text, where a 1-ulp difference can flip an argmax and diverge
the whole story. So it makes a good *numeric* regression signal.

`golden.json` is the reference value, computed on the host (trusted scalar
arithmetic). The firmware computes the same quantity on-device
(`CONFIG_LLM_EVAL`); CI fails if the device value drifts from golden by more
than `tolerance` (relative).

## Regenerating golden.json

`ppl.c` includes karpathy's `run.c` verbatim (its `main` is neutralized) so the
host math matches what `main/llm.c` ports. To recompute after changing the
model, tokenizer, or reference sentence:

```sh
curl -sSL -o run.c https://raw.githubusercontent.com/karpathy/llama2.c/master/run.c
cc -O2 -o ppl ppl.c -lm
./ppl ../data/stories260K.bin ../data/tok512.bin "One day, a clever fox found a shiny key under the old oak tree."
```

Then update `perplexity` (and `n_tokens`/`nll`) in `golden.json`. Updating the
golden value is a deliberate act — it redefines "correct", so it belongs in a
reviewed commit, never an automated one.

## The Q8_0 (int8) variant

`golden_q8.json` is the same eval for the group-wise int8 quantized model
(`data/stories260K_q8.bin`, produced by `scripts/export_q8.py`). CI builds a
second eval firmware with `CONFIG_LLM_MODEL_Q8=y` and checks the device value
against it, exercising the `forward_q8` path in `main/llm.c`.

`ppl_q.c` includes karpathy's `runq.c` verbatim (its main dropped via
`TESTING`) so the host math matches what `main/llm.c`'s int8 path ports — the
same relationship `ppl.c`/`run.c` have for fp32. To regenerate:

```sh
curl -sSL -o runq.c https://raw.githubusercontent.com/karpathy/llama2.c/master/runq.c
python ../scripts/export_q8.py ../data/stories260K.bin ../data/stories260K_q8.bin
cc -O2 -o ppl_q ppl_q.c -lm
./ppl_q ../data/stories260K_q8.bin ../data/tok512.bin "One day, a clever fox found a shiny key under the old oak tree."
```

The int8 golden carries a looser `tolerance` than the fp32 one for its first
hardware run; tighten it once the on-device value is known (int8 matmul
accumulates in exact int32, so device/host agreement should be tight).
