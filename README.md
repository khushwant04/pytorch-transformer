# PyTorch Transformer

A from-scratch PyTorch implementation of the encoder-decoder Transformer from
"Attention Is All You Need", trained for English-to-Italian translation on the `opus_books` dataset.

## What's here

All model components are written by hand in `model.py` (no `nn.Transformer`):

| Class | Role |
| --- | --- |
| `InputEmbeddings` | Token embedding scaled by `sqrt(d_model)` |
| `PositionalEncoding` | Fixed sinusoidal position encodings (registered buffer) |
| `MultiHeadAttentionBlock` | Scaled dot-product attention split across `h` heads, with masking |
| `FeedForwardBlock` | Position-wise `Linear -> ReLU -> Dropout -> Linear` |
| `LayerNormalization` | Layer norm with learnable gain and bias |
| `ResidualConnection` | Residual + dropout around each sub-layer |
| `EncoderBlock` / `Encoder` | Self-attention + FFN, stacked `N` times |
| `DecoderBlock` / `Decoder` | Masked self-attention + cross-attention + FFN, stacked `N` times |
| `ProjectionLayer` | Final linear layer to target-vocabulary logits |
| `build_transformer()` | Assembles the model and applies Xavier-uniform init |

Defaults follow the base model from the paper: `d_model=512`, `N=6` layers, `h=8` heads,
`d_ff=2048`, dropout `0.1`.

Other files:

- `dataset.py` - `BilingualDataset` adds `[SOS]`/`[EOS]`/`[PAD]` and builds the encoder padding mask
  and the decoder padding + causal mask (`causal_mask()`).
- `train.py` - word-level tokenizers (Hugging Face `tokenizers`), training loop (Adam, cross-entropy
  with label smoothing 0.1), greedy decoding, and validation with CER and BLEU from `torchmetrics`,
  logged to TensorBoard.
- `config.py` - hyperparameters and checkpoint path helpers.
- `tokenizer_en.json`, `tokenizer_it.json` - pre-built word-level tokenizers (rebuilt automatically if deleted).

Differences from the paper:

- Layer norm is applied **before** each sub-layer (pre-norm: `x + Dropout(Sublayer(LN(x)))`), whereas
  the paper applies it after the residual add. A final `LayerNormalization` follows the encoder and
  decoder stacks.
- Training uses a constant learning rate (`1e-4`) instead of the paper's warmup + inverse-sqrt schedule.
- Embeddings are not shared with the output projection.

## Project structure

```
.
├── config.py          # get_config(), weight-path helpers
├── dataset.py         # BilingualDataset, causal_mask
├── model.py           # Transformer components + build_transformer
├── train.py           # tokenizers, training loop, greedy decode, validation
├── tokenizer_en.json
├── tokenizer_it.json
└── requirements.txt
```

## Getting started

```bash
git clone https://github.com/khushwant04/pytorch-transformer.git
cd pytorch-transformer
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python train.py
```

`train.py` downloads `opus_books` (`en-it`) through Hugging Face `datasets`, splits it 90/10 into
train/validation, and trains for `num_epochs` (default 20). A CUDA GPU is used if available.

Outputs:

- Checkpoints: `opus_books_weights/tmodel_<epoch>.pt` (model, optimizer, epoch, global step).
- TensorBoard logs: `runs/tmodel` (train loss, validation CER and BLEU). View with `tensorboard --logdir runs`.
- After each epoch, two random validation sentences are greedily decoded and printed.

Resuming: `"preload": "latest"` in `config.py` resumes from the newest checkpoint in
`opus_books_weights/`. Set it to `None` to start fresh, or to an epoch string such as `"05"`.

## Configuration

Edit `get_config()` in `config.py`:

| Key | Default | Notes |
| --- | --- | --- |
| `batch_size` | 8 | |
| `num_epochs` | 20 | |
| `lr` | 1e-4 | Adam, `eps=1e-9` |
| `seq_len` | 350 | Every sentence must fit, otherwise `BilingualDataset` raises `Sentence is too long` |
| `d_model` | 512 | |
| `lang_src` / `lang_tgt` | `en` / `it` | Any language pair available in `opus_books` |

If you change the language pair, delete the existing `tokenizer_*.json` files so new ones are trained.

## References

- Vaswani et al., *Attention Is All You Need*, NeurIPS 2017. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- Ba, Kiros, Hinton, *Layer Normalization*, 2016. [arXiv:1607.06450](https://arxiv.org/abs/1607.06450)
- Szegedy et al., *Rethinking the Inception Architecture for Computer Vision* (label smoothing), 2015. [arXiv:1512.00567](https://arxiv.org/abs/1512.00567)
- Dataset: [OPUS Books](https://huggingface.co/datasets/Helsinki-NLP/opus_books) on Hugging Face.
