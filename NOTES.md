# Reading-Group Digest #1 — Attention Is All You Need

**Paper:** *Attention Is All You Need* (Vaswani et al.)
**arXiv ID:** 1706.03762

All numbers below were read from the paper's own text (abstract, Table 2, and the Results section) rather than from memory.

## WMT 2014 BLEU scores, as the paper gives them

- **English-to-German: 28.4 BLEU** — abstract: "Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU." Table 2 likewise reports 28.4 for Transformer (big) (EN-DE).
- **English-to-French: 41.8 BLEU** — abstract: "On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8." Table 2 likewise reports 41.8 for Transformer (big) (EN-FR).

For reference, Table 2 also gives the base model: 27.3 (EN-DE) and 38.1 (EN-FR).

> Internal-inconsistency note: the prose in the Results section (§ "Machine Translation") says "our big model achieves a BLEU score of 41.0" for EN-FR, which conflicts with the abstract and Table 2, both of which say 41.8. The abstract and Table 2 values are quoted above as the paper's headline figures.

## Big model: stated training time and GPU count

- **Training time:** 3.5 days (Results: "Training took 3.5 days on 8 P100 GPUs."; abstract: "after training for 3.5 days on eight GPUs")
- **GPU count:** 8 (P100 GPUs, one machine)

## What the architecture replaces

The Transformer is based solely on attention mechanisms and thereby dispenses with recurrence and convolutions entirely.

---

We will track our reproduction runs in Weights & Biases.
