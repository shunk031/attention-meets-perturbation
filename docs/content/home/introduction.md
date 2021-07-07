+++
# A Demo section created with the Blank widget.
# Any elements can be added in the body: https://sourcethemes.com/academic/docs/writing-markdown-latex/
# Add more sections by duplicating this file and customizing to your requirements.

widget = "blank"  # See https://sourcethemes.com/academic/docs/page-builder/
headless = true  # This file represents a page section.
active = true  # Activate this widget? true/false
weight = 1  # Order that this section will appear.

title = "Attention Meets Perturbations: Robust and Interpretable Attention with Adversarial Training"
subtitle = ""

[design]
  # Choose how many columns the section has. Valid values: 1 or 2.
  columns = "1"

[design.background]
  # Apply a background color, gradient, or image.
  #   Uncomment (by removing `#`) an option to apply it.
  #   Choose a light or dark text color by setting `text_color_light`.
  #   Any HTML color name or Hex value is valid.

  # Background color.
  # color = "navy"
  
  # Background gradient.
  # gradient_start = "DarkGreen"
  # gradient_end = "ForestGreen"
  
  # Background image.
  # image = "image.jpg"  # Name of image in `static/media/`.
  # image_darken = 0.6  # Darken the image? Range 0-1 where 0 is transparent and 1 is opaque.
  # image_size = "cover"  #  Options are `cover` (default), `contain`, or `actual` size.
  # image_position = "center"  # Options include `left`, `center` (default), or `right`.
  # image_parallax = true  # Use a fun parallax-like fixed background effect? true/false
  
  # Text color (true=light or false=dark).
  text_color_light = false

[design.spacing]
  # Customize the section spacing. Order is top, right, bottom, left.
  padding = ["20px", "0", "20px", "0"]

[advanced]
 # Custom CSS. 
 css_style = ""
 
 # CSS class.
 css_class = ""
+++

Shunsuke Kitada and Hitoshi Iyatomi

<div class="img_center">
  {{< readfile file="./static/media/figure1.svg" >}}
</div>

Attention mechanisms[^1] are widely applied in natural language processing (NLP) field through deep neural networks (DNNs). As the effectiveness of attention mechanisms became apparent in various tasks[^2] [^3] [^4] [^5] [^6] [^7], they were applied not only to recurrent neural networks (RNNs) but also to convolutional neural networks (CNNs). 
Moreover, Transformers [^8] which make proactive use of attention mechanisms have also achieved excellent results.
However, it has been pointed out that DNN models tend to be locally unstable, and even tiny perturbations to the original inputs [^9] or attention mechanisms can mislead the models~\cite{jain2019attention}.
Specifically, Jain and Wallace [^10] used a practical bi-directional RNN (BiRNN) model to investigate the effect of attention mechanisms and reported that learned attention weights based on the model are vulnerable to perturbations.

To tackle the models' vulnerability to perturbation, Goodfellow et al.[^11] proposed adversarial training (AT) that increases robustness by adding adversarial perturbations to the input and the training technique forcing the model to address its difficulties. 
Previous studies[^11] [^12] in the image recognition field have theoretically explained the regularization effect of AT and shown that it improves the robustness of the model for unseen images.

In this paper, we propose a new general training technique for attention mechanisms based on AT, called `adversarial training for attention` (**Attention AT**) and `more interpretable adversarial training for attention` (**Attention iAT**). 
The proposed techniques are the first attempt to employ AT for attention mechanisms.
The proposed Attention AT/iAT is expected to improve the robustness and the interpretability of the model by appropriately overcoming the adversarial perturbations to attention mechanisms [^13] [^14] [^15].
Because our proposed AT techniques for attention mechanisms is model-independent and a general technique, it can be applied to various DNN models (e.g., RNN and CNN) with attention mechanisms.
Our technique can also be applied to any similarity functions for attention mechanisms, e.g, additive function [^1] and scaled dot-product function [^8], which is famous for calculating the similarity in attention mechanisms.

To demonstrate the effects of these techniques, we evaluated them compared to several other state-of-the-art AT-based techniques [^16] [^17] with ten common datasets for different NLP tasks. 
These datasets included binary classification (BC), question answering (QA), and natural language inference (NLI).
We also evaluated how the attention weights obtained through the proposed AT technique agreed with the word importance calculated by the gradients~\cite{simonyan2013deep}. 
Evaluating the proposed techniques, we obtained the following findings concerning AT for attention mechanisms in NLP:
- AT for attention mechanisms improves the prediction performance of various NLP tasks.
- AT for attention mechanisms helps the model learn cleaner attention and demonstrates a stronger correlation with the word importance calculated from the model gradients.
- The proposed training techniques are much less independent concerning perturbation size in AT.

Especially, our Attention iAT demonstrated the best performance in nine out of ten tasks and more interpretable attention, i.e., resulting attention weight correlated more strongly with the gradient-based word importance.

{{<image src="./media/logo_hosei.png" alt="Hosei University" max_width="30">}}

[^1]: D. Bahdanau, K. Cho, and Y. Bengio, “Neural machine translation by jointly learning to align and translate,” CoRR preprint arXiv:1409.0473,2014.

[^2]: Z. Lin, M. Feng, C. N. dos Santos, M. Yu, B. Xiang, B. Zhou, and Y. Bengio,  “A structured self-attentive sentence embedding,”  in Proc. of the 5th International Conference on Learning Representations, ICLR, Conference Track Proceedings, 2017.

[^3]: Y. Wang, M. Huang, and L. Zhao, “Attention-based LSTM for aspect-level  sentiment classification,” in Proc. of the 2016 Conference on Empirical Methods in Natural Language Processing, ser. Associationfor Computational Linguistics (ACL), 2016, pp. 606–615.

[^4]: X. He and D. Golub, “Character-level question answering with attention,” in Proc. of the 2016 Conference on Empirical Methods in Natural Language Processing ser. Association for Computational Linguistics (ACL), 2016, pp. 1598–1607.

[^5]: A. Parikh, O. Täckström, D. Das, and J. Uszkoreit, “A decomposable attention  model for natural language inference,” in Proc. of the 2016 Conference on Empirical Methods in Natural Language Processing, ser.Association for Computational Linguistics (ACL), 2016, pp. 2249–2255.

[^6]: T. Luong, H. Pham, and C. D. Manning, “Effective approaches to attention-based neural machine translation,” in Proc. of the 2015 Conference on Empirical Methods in Natural Language Processing, ser. Association for Computational Linguistics (ACL), 2015, pp. 1412–1421.

[^7]: A. M. Rush, S. Chopra, and J. Weston, “A neural attention model for abstractive sentence summarization,” in Proc. of the 2015 Conference on Empirical Methods in Natural Language Processing, ser. Associationfor Computational  Linguistics (ACL), 2015, pp. 379–389.

[^8]: A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N.Gomez, Ł Kaiser, and I. Polosukhin, “Attention is all you need,” in Proc. of the 30th International Conference on Neural Information Processing Systems, 2017, pp 5998–6008.

[^9]: C. Szegedy, W. Zaremba, I. Sutskever, J. Bruna, D. Erhan, I. Goodfellow, and R. Fergus, “Intriguing properties of neural networks,” in  2nd International Conference on Learning Representations, ICLR, Conference Track Proceedings, 2013.

[^10]: S. Jain and B. C. Wallace, “Attention is not explanation,” in Proc. of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), ser. Association for Computational Linguistics (ACL), 2019, pp. 3543–3556.

[^11]: I. J. Goodfellow, J. Shlens, and C. Szegedy, “HExplaining and harnessing adversarial examples,” in 3rd International Conference on Learning Representations, ICLR, Conference  Track Proceedings, 2014.

[^12]: U. Shaham, Y. Yamada, and S. Negahban, “Understanding adversarial training: Increasing local stability of supervised models through robust optimization,” Neurocomputing,  vol. 307, pp. 195–204, 2018.

[^13]: D. Tsipras, S. Santurkar, L. Engstrom, A. Turner, and A. Madry, “Robustness may be at odds with accuracy,” in Proc. of the International Conference on Learning Representations, ICLR, 2019.

[^14]: T. Itazuri, Y. Fukuhara, H. Kataoka, and S. Morishima, “What  doadversarially robust models look at?” CoRR preprint arXiv:1905.07666,2019.

[^15]: T. Zhang and Z. Zhu, “Interpreting adversarially trained convolutional neural networks,” in International Conference on Machine Learning. PMLR, 2019, pp. 7502–7511.

[^16]: T. Miyato, A. M. Dai, and I. Goodfellow, “Adversarial training methods for semi-supervised text classification,” in Proc. of the 5th International Conference on Learning  Representations, ICLR, Conference Track Proceedings, 2016.

[^17]: M. Sato, J. Suzuki, H. Shindo, and Y. Matsumoto, “Interpretable adversarial perturbation in input embedding space for text,”  in Proc. of the 27th International Joint Conference on Artificial Intelligence, ser. AAAI Press, 2018, pp. 4323–4330.

[^20]: K. Simonyan, A. Vedaldi, and A. Zisserman, “Deep inside convolutional networks:  Visualising image classification models and saliency  maps,”in Proc. of the 2nd International Conference on Learning Representations, ICLR, Workshop Track Proceedings, 2013.
