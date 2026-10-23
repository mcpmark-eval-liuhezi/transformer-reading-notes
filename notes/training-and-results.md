# Training and Results — Attention Is All You Need (arXiv:1706.03762)

All figures below are taken directly from the paper's abstract.

## Headline results

- **WMT 2014 English-to-German:** 28.4 BLEU. This improves over the existing best results, **including ensembles, by over 2 BLEU**.
- **WMT 2014 English-to-French:** 41.8 BLEU. This is a **new single-model state-of-the-art** BLEU score.

## Training cost

- **Training time:** 3.5 days
- **Hardware:** eight GPUs (trained on 8 GPUs)
- The abstract notes this is "a small fraction of the training costs of the best models from the literature."

## Other facts from the abstract

- The Transformer is based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.
- Experiments on two machine translation tasks show these models are superior in quality while being more parallelizable and requiring significantly less time to train.
- The Transformer generalizes well to other tasks, demonstrated by successful application to English constituency parsing both with large and limited training data.
