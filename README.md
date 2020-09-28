# Attention Meets Perturbation: Robust and Interpretable Attention with Adversarial Training

[![](http://img.shields.io/badge/cs.AI-arXiv%3A2009.12064-B31B1B.svg)](http://arxiv.org/abs/2009.12064)
![Python 3.7](https://img.shields.io/badge/python-3.7%2B-brightgreen.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Powered by AllenNLP](https://img.shields.io/badge/Powered%20by-AllenNLP-blue.svg)](https://github.com/allenai/allennlp)

| | |
|------|------|
| ![model](./.github/assets/BC-model.png)| ![Figure 1](./.github/assets/figure1.png) |

**Attention Meets Perturbation: Robust and Interpretable Attention with Adversarial Training**  
Shunsuke Kitada and Hitoshi Iyatomi

Preprint: https://arxiv.org/abs/2009.12064

Abstract: *In recent years, deep learning models have placed more emphasis on the interpretability and robustness of models. The attention mechanism is an important technique that contributes to these elements and is widely used, especially in the natural language processing (NLP) field. Adversarial training (AT) is a powerful regularization technique for enhancing the robustness of neural networks and has been successful in many applications. The application of AT to the attention mechanism is expected to be highly effective, but there is little research on this. In this paper, we propose a new general training technique for NLP tasks, using AT for attention (Attention AT) and more interpretable adversarial training for attention (Attention iAT). Our proposals improved both the prediction performance and interpretability of the model by applying AT to the attention mechanisms. In particular, Attention iAT enhances those advantages by introducing adversarial perturbation, which differentiates the attention of sentences where it is unclear which words are important. We performed various NLP tasks on ten open datasets and compared the performance of our techniques to a recent model using attention mechanisms. Our experiments revealed that AT for attention mechanisms, especially Attention iAT, demonstrated (1) the best prediction performance in nine out of ten tasks and (2) more interpretable attention (i.e., the resulting attention correlated more strongly with gradient-based word importance) for all tasks. Additionally, our techniques are (3) much less dependent on perturbation size in AT.*

## Note

This paper is under review. Source code will be revealed upon acceptance.

## Citation

```
@article{kitada2020attention,
  title   = {Attention Meets Perturbation: Robust and Interpretable Attention with Adversarial Training},
  author  = {Shunsuke Kitada and Hitoshi Iyatomi},
  journal = {CoRR},
  volume  = {abs/2009.12064},
  year    = {2020},
}
```

## Reference

- Kitada, Shunsuke, and Hitoshi Iyatomi. "Attention Meets Perturbation: Robust and Interpretable Attention with Adversarial Training" CoRR preprint arXiv:2009.12064 (2020).
