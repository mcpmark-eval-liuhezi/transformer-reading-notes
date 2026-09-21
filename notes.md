# Attention Is All You Need — Reading Group Cheat Sheet

**Paper:** Vaswani et al., *Attention Is All You Need* (arXiv:1706.03762, "Transformer")
**Session:** Thursday — sequence transduction, paper 1 of the series.

---

## TL;DR

The Transformer replaces recurrence and convolutions **entirely** with attention. It is an
encoder–decoder stack made of multi-head self-attention plus position-wise feed-forward layers,
trained on WMT 2014 translation, and it beats the previous state of the art on both tasks at a
fraction of the training cost.

---

## The Three Things You Must Remember

### 1. Headline WMT 2014 translation results (BLEU, newstest2014)

| Model | EN→DE | EN→FR |
|---|---|---|
| **Transformer (big)** | **28.4** | **41.8** |
| Transformer (base) | 27.3 | 38.1 |

- **EN→German: 28.4 BLEU** — beats the best previously reported models *including ensembles*
  (GNMT+RL Ensemble 26.30, ConvS2S Ensemble 26.36) by **more than 2.0 BLEU**.
- **EN→French: 41.8 BLEU** — a new **single-model** state of the art, at less than 1/4 the
  training cost of the previous state of the art.
- Note the base model is already competitive: 27.3 / 38.1 BLEU for a small fraction of the FLOPs.

### 2. Word order without recurrence — positional encodings

Because the model has **no recurrence and no convolution**, it has no built-in notion of token
order, so order information must be injected explicitly:

- **What is added:** "positional encodings" are **added to the input embeddings** at the
  **bottoms of both the encoder and the decoder stacks** (element-wise sum, not concatenation).
- **Dimension:** the positional encodings have the **same dimension $d_{model}$** as the embeddings
  (512 in the base model), so the two can simply be summed.
- **Which functions the base model uses:** **fixed sinusoidal** functions — sine and cosine of
  different frequencies:

$$PE_{(pos,2i)} = \sin(pos / 10000^{2i/d_{model}})$$
$$PE_{(pos,2i+1)} = \cos(pos / 10000^{2i/d_{model}})$$

  where $pos$ is the position and $i$ is the dimension; wavelengths form a geometric progression
  from $2\pi$ to $10000 \cdot 2\pi$.
- **Why sinusoids:** for any fixed offset $k$, $PE_{pos+k}$ is a linear function of $PE_{pos}$,
  which should make *relative* positions easy to learn; it may also let the model extrapolate to
  sequences longer than any seen in training.
- **Learned vs. fixed:** the authors also tried **learned positional embeddings** and found
  **nearly identical results** (Table 3, row (E): 25.7 vs. 25.8 BLEU on newstest2013). They kept
  the sinusoidal version for the extrapolation argument.

### 3. Training cost in hardware terms

- **GPUs:** one machine with **8 NVIDIA P100 GPUs**.
- **Big model:** trained for **300,000 steps = 3.5 days** (1.0 s per step). → $2.3\cdot10^{19}$ FLOPs.
- **Base model:** 100,000 steps = **12 hours** (0.4 s per step). → $3.3\cdot10^{18}$ FLOPs.
- Training cost is estimated as training time × number of GPUs × sustained single-precision FLOPS
  (9.5 TFLOPS assumed for a P100; 2.8/3.7/6.0 for K80/K40/M40).

---

## Architecture at a Glance (context)

- **Encoder:** stack of $N=6$ identical layers; each layer = multi-head self-attention + position-wise
  feed-forward network (FFN with $d_{ff}=2048$). Residual connection + layer normalization around each
  sub-layer: $\mathrm{LayerNorm}(x + \mathrm{Sublayer}(x))$.
- **Decoder:** also $N=6$ layers, with a **third** sub-layer performing multi-head attention over the
  encoder output (encoder–decoder attention). Self-attention is **masked** so positions cannot attend
  to subsequent positions, keeping generation auto-regressive.
- **Attention core:** scaled dot-product attention,
  $\mathrm{Attention}(Q,K,V) = \mathrm{softmax}(QK^T/\sqrt{d_k})V$; scaling by $1/\sqrt{d_k}$ keeps
  the softmax out of tiny-gradient regions for large $d_k$. Multi-head: $h=8$ heads with
  $d_k = d_v = 64$ in the base model, run in parallel and concatenated.
- **Training data:** WMT 2014 EN-DE ≈ 4.5M sentence pairs (BPE, ~37k shared vocab);
  WMT 2014 EN-FR ≈ 36M sentences (32k word-piece vocab). Batches of ~25k source + 25k target tokens.
- **Optimizer/regularization:** Adam ($\beta_1=0.9, \beta_2=0.98, \epsilon=10^{-9}$) with the
  warmup schedule ($warmup\_steps=4000$); residual dropout $P_{drop}=0.1$ (0.3 for the big EN→FR
  model) applied to sub-layer outputs **and to the sums of embeddings and positional encodings**;
  label smoothing $\epsilon_{ls}=0.1$.
- **Decoding:** beam search, beam size 4, length penalty $\alpha=0.6$; checkpoints averaged
  (last 5 for base, last 20 for big).

---

## Good Discussion Questions

1. The FLOPs comparison mixes different GPUs and "sustained capacity" assumptions — how fair is that?
2. Sinusoidal vs. learned positional encodings gave nearly identical BLEU — which would you pick today, and why?
3. Is "attention is all you need" still literally true, given that the FFN layers carry a lot of the parameters?
4. Dropout is applied *on the sum of embeddings + positional encodings* — what is that protecting against?

---

*All figures and claims above were checked against the LaTeX source of arXiv:1706.03762
(Table 2 / abstract for BLEU, §3.5 for positional encodings, §5.3 for hardware, Table 3 for
architecture hyper-parameters). Please comment directly on this file / the PR with corrections —
it will be merged before Thursday's session.*
