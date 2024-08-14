---
title: Image-to-Text
emoji: 🏃
colorFrom: gray
colorTo: red
sdk: gradio
sdk_version: "4.39.0"
app_file: app.py
pinned: false
---

# Image-to-Text

Image Captioning using ResNet50 and LSTM in Pytorch

### Model

- ResNet50 - Extract image feature (pretrained) as encoder
- LSTM - Generate descripton of image from extracted feature as decoder
- BERT tokenizer - Tokenize the label and detokenize the LSTM perdiction (pretrained)

![alt text](model-arch.png)

### Training

- Train on flickr30k dataset
- Train on GPU P100 on kaggle 
- Number of Epochs = 8

### Result

- Training loss = 0.3914
- Validation loss = 0.3942
- Average BLEU Score = 0.4112

![alt text](loss.png)

### Deployement

- Built UI interface using Gradio  
- Deployed on Hugging Face Gradio spaces
