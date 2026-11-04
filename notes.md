# Reading-group cheat sheet #1 — *Attention Is All You Need*

**Paper:** Vaswani et al., *Attention Is All You Need* (arXiv:1706.03762)
**Source for fact-checking:** the paper's LaTeX source (abstract, §3.5, §5.3, Table 2 / §6.1 results section)

## 1. WMT 2014 headline BLEU results (verified: abstract + Table 2)

- **English → German (newstest2014): 28.4 BLEU** — the big model, a new state of the art, beating the previous best results (including ensembles) by more than 2 BLEU.
- **English → French (newstest2014): 41.8 BLEU** — a new **single-model** state of the art.
- For context: the base model scores 27.3 (EN→DE) and 38.1 (EN→FR) at a much lower training cost (~3.3×10¹⁸ FLOPs). The big model's estimated cost is ~2.3×10¹⁹ FLOPs — a small fraction of the best previous models (e.g. ~10²⁰ for the GNMT+RL ensemble EN→FR).

> Discussion note: the Results prose (§6.1) quotes 41.0 BLEU for EN→FR, which conflicts with the abstract and Table 2 (41.8); the paper's headline claim is 41.8.

## 2. Word order with no recurrence (§3.5 — Positional Encoding)

- The Transformer has **no recurrence and no convolution**, so it cannot itself know token order. It compensates by **adding "positional encodings" to the input embeddings at the bottoms of the encoder and decoder stacks** — i.e., the same dimension as the embeddings (`d_model`), so the two can simply be summed.
- The **base model uses fixed sinusoidal functions** of different frequencies:

  $PE_{(pos,2i)} = sin(pos / 10000^{2i/d_{model}})$, $PE_{(pos,2i+1)} = cos(pos / 10000^{2i/d_{model}})$

  i.e., each dimension of the positional encoding corresponds to a sinusoid, with wavelengths forming a geometric progression from $2\pi$ to $10000 \cdot 2\pi$. The choice is motivated by the guess that it would allow the model to easily learn to attend **by relative position**: for any fixed offset $k$, $PE_{pos+k}$ can be represented as a linear function of $PE_{pos}$.
- The sinusoidal version was chosen over **learned positional embeddings** (Table 3, row E), because the two produced nearly identical results AND the sinusoids may extrapolate to sequence lengths longer than any seen during training.

## 3. Training cost in hardware terms (§5.3 — Hardware and Schedule)

- All models were trained on **one machine with 8 NVIDIA P100 GPUs**.
- The **big models took 3.5 days to train** (300,000 steps at ~1.0 s/step).
- For reference, the base models took 12 hours (100,000 steps at ~0.4 s/step).

## Architecture in one line

Stack of 6 identical encoder layers and 6 identical decoder layers over (scaled dot-product) multi-head self-attention + encoder-decoder attention and position-wise feed-forward networks ($d_{model} = 512$ base, $h = 8$ heads).

## Checklist for the group

- [ ] BLEU numbers (28.4 EN→DE, 41.8 EN→FR) match Table 2
- [ ] Positional-encoding description matches §3.5
- [ ] Hardware/time description matches §5.3
- [ ] Anything to add for first-timers to sequence transduction?

_Subject to your review and corrections — please comment on this PR or directly on `notes.md` before Thursday._
