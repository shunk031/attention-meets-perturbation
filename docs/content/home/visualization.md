+++
# A Demo section created with the Blank widget.
# Any elements can be added in the body: https://sourcethemes.com/academic/docs/writing-markdown-latex/
# Add more sections by duplicating this file and customizing to your requirements.

widget = "blank"  # See https://sourcethemes.com/academic/docs/page-builder/
headless = true  # This file represents a page section.
active = true  # Activate this widget? true/false
weight = 1  # Order that this section will appear.

title = "Visualization"
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

The following tables are visualizations of the attention weight for each word and gradient-based word importance in the [Stanford Sentiment Treebank (SST)](https://nlp.stanford.edu/sentiment/index.html) [^1] test dataset.

## Vanilla vs. Attention AT vs. Attention iAT
Attention AT yielded clearer attention compared to the Vanilla model or Attention iAT.
Specifically, Attention AT tended to strongly focus attention on a few words.

{{< readfile file="./static/media/attention_models.html" markdown="true">}}

## Attention and Gradient
Regarding the correlation of word importance based on attention weights and gradient-based word importance, Attention iAT demonstrated higher similarities than other models.

### Vanilla
{{< readfile file="./static/media/attention_gradient_vanilla.html" markdown="true">}}

### Attention AT
{{< readfile file="./static/media/attention_gradient_AttentionAT.html" markdown="true">}}

### Attention iAT
{{< readfile file="./static/media/attention_gradient_AttentioniAT.html" markdown="true">}}

[^1]: R. Socher, A. Perelygin, J. Wu, J. Chuang, C. D. Manning, A. Ng, and C. Potts, “Recursive deep models for semantic compositionality over a sentiment treebank,” in Proc. of the 2013 Conference on Empirical Methods in Natural Language Processing, ser. Association for Computational Linguistics (ACL), 2013, pp. 1631–1642.
