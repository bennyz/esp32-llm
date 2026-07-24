/* Host-side perplexity for the Q8_0 (int8) quantized model.
 * Includes karpathy's runq.c verbatim (TESTING drops its main) so the
 * arithmetic here is the trusted reference the firmware's int8 path ports.
 * Mirror of ppl.c, which does the same for the fp32 run.c. */
#define TESTING
#include "runq.c"

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
    char *checkpoint = "stories260K_q8.bin";
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
