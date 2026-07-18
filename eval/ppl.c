/* Host-side golden perplexity for the ESP32 eval.
 * Includes karpathy's run.c verbatim (its main is neutralized) so the
 * arithmetic here is identical to what the firmware ports. */
#define main run_c_main_unused
#include "run.c"
#undef main

/* Teacher-forced perplexity over a fixed reference sentence.
 * Must stay identical to eval_perplexity() in main/llm.c. */
static float eval_perplexity(Transformer *t, Tokenizer *tok, char *text) {
    int vocab_size = t->config.vocab_size;
    int *tokens = malloc((strlen(text) + 3) * sizeof(int));
    int n = 0;
    encode(tok, text, /*bos*/1, /*eos*/0, tokens, &n);
    if (n < 2) { free(tokens); return -1.0f; }

    double nll = 0.0;
    for (int pos = 0; pos < n - 1; pos++) {
        float *logits = forward(t, tokens[pos], pos);
        softmax(logits, vocab_size);
        float p = logits[tokens[pos + 1]];
        nll += -log((double)p);
    }
    float ppl = (float)exp(nll / (n - 1));
    fprintf(stderr, "tokens=%d nll=%.6f\n", n, nll);
    for (int i = 0; i < n; i++) fprintf(stderr, "%d ", tokens[i]);
    fprintf(stderr, "\n");
    free(tokens);
    return ppl;
}

int main(int argc, char **argv) {
    char *checkpoint = "stories260K.bin";
    char *tokenizer = "tok512.bin";
    char *text = "Once upon a time, there was a little girl named Lily.";
    if (argc > 1) checkpoint = argv[1];
    if (argc > 2) tokenizer = argv[2];
    if (argc > 3) text = argv[3];

    Transformer transformer;
    build_transformer(&transformer, checkpoint);
    Tokenizer tok;
    build_tokenizer(&tok, tokenizer, transformer.config.vocab_size);

    float ppl = eval_perplexity(&transformer, &tok, text);
    printf("perplexity: %.6f\n", ppl);

    free_tokenizer(&tok);
    free_transformer(&transformer);
    return 0;
}
