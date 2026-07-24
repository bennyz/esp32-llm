#ifndef LLM_H
#define LLM_H

/**
 * Original author of this:
 * https://github.com/karpathy/llama2.c 
 * 
 * Slight modifications added to make it ESP32 friendly
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include "freertos/task.h"
#include "freertos/event_groups.h"

typedef float v4sf __attribute__((aligned(16)));

// A group-wise int8 quantized tensor (Q8_0): `q` holds the int8 values, `s`
// holds one fp32 scale per group of GS values. Used only for quantized
// checkpoints (magic "ak42", version 2); the fp32 path leaves these NULL.
typedef struct {
    int8_t *q; // quantized values
    float *s;  // scaling factors, one per group
} QuantizedTensor;

typedef struct {
    float prob;
    int index;
} ProbIndex; // struct used when sorting probabilities during top-p sampling

typedef struct {
    int vocab_size;
    ProbIndex* probindex; // buffer used in top-p sampling
    float temperature;
    float topp;
    unsigned long long rng_state;
} Sampler;

typedef struct {
    char *str;
    int id;
} TokenIndex;

typedef struct {
    char** vocab;
    v4sf* vocab_scores;
    TokenIndex *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512]; // stores all single-byte strings
} Tokenizer;

typedef struct {
    int dim; // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers; // number of layers
    int n_heads; // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len; // max sequence length
} Config;

typedef struct {
    // token embedding table
    v4sf* token_embedding_table;    // (vocab_size, dim)
    // weights for rmsnorms
    v4sf* rms_att_weight; // (layer, dim) rmsnorm weights
    v4sf* rms_ffn_weight; // (layer, dim)
    // weights for matmuls. note dim == n_heads * head_size
    v4sf* wq; // (layer, dim, n_heads * head_size)
    v4sf* wk; // (layer, dim, n_kv_heads * head_size)
    v4sf* wv; // (layer, dim, n_kv_heads * head_size)
    v4sf* wo; // (layer, n_heads * head_size, dim)
    // weights for ffn
    v4sf* w1; // (layer, hidden_dim, dim)
    v4sf* w2; // (layer, dim, hidden_dim)
    v4sf* w3; // (layer, hidden_dim, dim)
    // final rmsnorm
    v4sf* rms_final_weight; // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    v4sf* wcls;

    // --- quantized (Q8_0) weights, used only when Transformer.quantized != 0 ---
    // The fp32 pointers above are unused in that mode, except
    // token_embedding_table which is populated with the dequantized table.
    QuantizedTensor *q_tokens; // (vocab_size, dim)
    QuantizedTensor *qwq, *qwk, *qwv, *qwo; // (layer, ...)
    QuantizedTensor *qw1, *qw2, *qw3;       // (layer, ...)
    QuantizedTensor *qwcls;                 // shared with q_tokens if tied
} TransformerWeights;

typedef struct {
    // current wave of activations
    v4sf *x; // activation at current time stamp (dim,)
    v4sf *xb; // same, but inside a residual branch (dim,)
    v4sf *xb2; // an additional buffer just for convenience (dim,)
    v4sf *hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    v4sf *hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    v4sf *q; // query (dim,)
    v4sf *k; // key (dim,)
    v4sf *v; // value (dim,)
    v4sf *att; // buffer for scores/attention values (n_heads, seq_len)
    v4sf *logits; // output logits
    // quantized activation buffers (Q8_0 path only)
    QuantizedTensor xq; // quantized x (dim,)
    QuantizedTensor hq; // quantized hb (hidden_dim,)
    // kv cache
    v4sf* key_cache;   // (layer, seq_len, dim)
    v4sf* value_cache; // (layer, seq_len, dim)
} RunState;


typedef struct {
    Config config; // the hyperparameters of the architecture (the blueprint)
    TransformerWeights weights; // the weights of the model
    RunState state; // buffers for the "wave" of activations in the forward pass
    // some more state needed to properly clean up the memory mapping (sigh)
    int fd; // file descriptor for memory mapping
    v4sf* data; // memory mapped data pointer
    size_t file_size; // size of the checkpoint file in bytes
    int quantized; // 1 if the checkpoint is Q8_0 (magic "ak42", version 2)
    int group_size; // quantization group size (GS), quantized checkpoints only
} Transformer;



typedef void (*generated_complete_cb)(float tokens_ps);

void build_transformer(Transformer *t, char* checkpoint_path);
void build_tokenizer(Tokenizer* t, char* tokenizer_path, int vocab_size);
void build_sampler(Sampler* sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed);
void generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int steps, generated_complete_cb cb_done);
float eval_perplexity(Transformer *t, Tokenizer *tok, char *text);
void free_sampler(Sampler* sampler);
void free_transformer(Transformer* t);
void free_tokenizer(Tokenizer* t);


#endif