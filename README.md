# Unigram Tokenization
This is a simple Python implementation of Unigram Tokenization.
## Algorithm Description
1. Use Byte Pair Encoding (BPE) tokenizer to create an arbitrarily large vocabulary $\mathcal{V}$.
2. Let the distribution on tokens be denoted as $p(x_i) = \frac{count(x_i)}{\sum_{j=1}^{N}count(x_j)}$.
3. Use the hard EM algorithm to estimate the distribution $p(x_i)$:
    - Repeat these steps until convergence
        - Employ the Viterbi algorithm to find the best tokenization $\mathcal{T}$.
        - Fix the best tokenization $\mathcal{T}$ and maximize likelihood: 
        $$ P(x) = \prod_{i=1}^{N_{\mathcal{T}}} p(x_i)$$
4. Shrink the vocabulary $\mathcal{T}$ by a multiplication factor $\alpha$:
    - Calculate the loss if token $x_i$ is replaced with the Viterbi path of token ${x_i}$.
    - Sort by loss
    - Shrink the vocabulary so that $|\mathcal{T_{new}}| = (1 - \alpha)|\mathcal{T}_{old}|$.
## Source code
 - You can find Unigram tokenization realization in unigram.ipynb
## References
- Unigram Tokenizer. Insanely good article. https://everdark.github.io/k9/notebooks/ml/natural_language_understanding/subword_units/subword_units.nb.html
- Unigram Tokenizer. Rust realization. Some theoretical aspects. https://guillaume-be.github.io/2020-05-30/sentence_piece
- Sentencepiece library. C realization https://github.com/google/sentencepiece/blob/master/src/unigram_model_trainer.cc
- Unigram Tokenizer HuggingFace. https://huggingface.co/learn/nlp-course/en/chapter6/7?fw=pt