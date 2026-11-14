# Results: team salman

The score is the SST-2 test accuracy of the fixed linear probe (`microgpt/probe.py`, unchanged). All choices were made on **dev**.

| Step | Change | dev | **test** |
|---|---|---|---|
| 0 | Defaults (char tokens, 1×128, ctx 32, SST only) | — | 52.0% (README) |
| 1 | BPE 8k, ctx 64, 2×128, IMDB 0.8 + SST 0.2, 1,500 steps (1.7 min) | 64.4% | 61.2% |
| 2 | + richer featurizer (`my_model.py`: every layer's last-token and mean-over-tokens vectors) | 69.4% | 65.1% |
| 3 | + 4,500 more steps of the same model (`init_from`, 4.9 min), 3× more text | 70.0% | **66.0%** |

Step 1 changed several settings at once, so its gain is their combined effect. Steps 2 and 3 each changed one thing.

## Why it works
- **The defaults read only the last 31 characters of each sentence.** A 64-token BPE window reads nearly every SST sentence in full.
- **More text, more sentiment.** IMDB's training reviews have 57.9M characters, against 0.7M in SST's training sentences (about 83×). Predicting the next word of a review well requires tracking whether the review is positive.
- **Sentiment is spread over layers and positions.** The default probe reads one vector: the last token of the last layer. Adding mean-pooled and middle-layer vectors gave +4 points, and the L1 probe keeps only what helps.

## Reproduce
```
python -m microgpt.train backend=torch data.tokenizer=bpe data.vocab_size=8000 model.num_layers=2 \
  model.hidden_dim=128 model.num_heads=4 model.max_seq_len=64 model.tie_word_embeddings=true \
  data.weights.imdb=0.8 data.weights.sst=0.2 optimizer.lr_schedule=cosine optimizer.warmup=0.02 \
  threads=0 max_steps=1500 log_dir=runs/quick
python -m microgpt.train backend=torch init_from=runs/quick/model.safetensors data.weights.imdb=0.8 \
  data.weights.sst=0.2 optimizer.lr=0.002 optimizer.lr_schedule=cosine optimizer.warmup=0.02 threads=0 \
  num_epochs=100 max_steps=4500 log_dir=runs/final
CHECKPOINT=runs/final/model.safetensors python -m scripts.linear_probe model=my_model.featurize
```
