+++
# A Demo section created with the Blank widget.
# Any elements can be added in the body: https://sourcethemes.com/academic/docs/writing-markdown-latex/
# Add more sections by duplicating this file and customizing to your requirements.

widget = "blank"  # See https://sourcethemes.com/academic/docs/page-builder/
headless = true  # This file represents a page section.
active = true  # Activate this widget? true/false
weight = 4  # Order that this section will appear.

title = "Conclusion"
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

We proposed robust and interpretable attention training techniques that exploit AT.
In the experiments with various NLP tasks, we confirmed that AT for attention mechanisms achieves better performance than techniques using AT for word embedding in terms of the prediction performance and the interpretability of the model.
Specifically, the Attention iAT technique introduced adversarial perturbations that emphasized differences in the importance of words in a sentence and combined high accuracy with interpretable attention, which was more strongly correlated with the gradient-based method of word importance.
The proposed technique could be applied to various models and NLP tasks.
This paper provides strong support and motivation for utilizing AT with attention mechanisms in NLP tasks.

In the experiment, we demonstrated the effectiveness of the proposed techniques for RNN models that are reported to be vulnerable to attention mechanisms, but we will confirm the effectiveness of the proposed technique for large language models with attention mechanisms such as Transforme [^1] or BERT [^2] in the future.
Because the proposed techniques are model-independent and general techniques for attention mechanisms, we can expect they will improve predictability and the interpretability for language models.

[^1]: A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N.Gomez, Ł Kaiser, and I. Polosukhin, “Attention is all you need,” in Proc. of the 30th International Conference on Neural Information Processing Systems, 2017, pp 5998–6008.

[^2]: J. Devlin, M. W. Chang, K. Lee, and K. Toutanova, “Bert:   Pre-training of deep bidirectional transformers for language understanding,” in Proc. of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), ser Association for Computational Linguistics (ACL), 2019, pp. 4171–4186.
