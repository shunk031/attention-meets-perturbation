+++
# A Demo section created with the Blank widget.
# Any elements can be added in the body: https://sourcethemes.com/academic/docs/writing-markdown-latex/
# Add more sections by duplicating this file and customizing to your requirements.

widget = "blank"  # See https://sourcethemes.com/academic/docs/page-builder/
headless = true  # This file represents a page section.
active = true  # Activate this widget? true/false
weight = 1  # Order that this section will appear.

title = "Common Model Architectures"
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

 Illustration of the base models to apply our proposed training technique: (a) a single sequence model for the binary classification (BC) task and (b) a pair sequence model for question answering (QA) and natural language inference (NLI) tasks. In (a), the input of the model is word embeddings, {${\bf w_1}$, $\cdots$, ${\bf w_{T_S}}$} associated with the input sentence $X_S$. 
 In (b), the inputs are word embeddings {${\bf w_{1}^{(p)}}$, $\cdots$, ${\bf w_{T_P}^{(p)}}$} and {${\bf w_{1}^{(q)}}$, $\cdots$, ${\bf w_{T_Q}^{(q)}}$}  from two input sequences, $X_P$ and $X_Q$, respectively. 
 These inputs are encoded into hidden states through a bi-directional RNN (BiRNN) model. 
 In conventional models, perturbation ${\bf r}$ is added to the hidden state of the words ${\bf h}$. 
 In our adversarial training for attention mechanisms, we compute and add the worst-case perturbation ${\bf r}$ to attention ${\bf a}$ to improve both prediction performance and the interpretability of the model.

| (a) Single sequence model                             | (b) Pair sequence model                             |
|-------------------------------------------------------|-----------------------------------------------------|
| {{< readfile file="./static/media/single-seq.svg" >}} | {{< readfile file="./static/media/pair-seq.svg" >}} |
