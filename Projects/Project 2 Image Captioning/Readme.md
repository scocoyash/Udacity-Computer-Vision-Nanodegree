### Project Overview

----

[![Udacity Computer Vision Nanodegree](http://tugan0329.bitbucket.io/imgs/github/cvnd.svg)](https://www.udacity.com/course/computer-vision-nanodegree--nd891)<br/>

**Layman Terms Explanation:** Give an image, it will suggest you a caption(or whatever it understands) from the image.

This project contains an Image-Captioning Architecture studied and developed during the nanodegree project.
Image Captioning is based on Encoder-Decoder architecture where the CNN acts an encoder and LSTM based RNN acts as a decoder.

**Architecture of the network**
![Architecture developed](https://raw.githubusercontent.com/udacity/CVND---Image-Captioning-Project/master/images/encoder-decoder.png)

**Summary Of the project:**
- Dataset used: COCO by Microsoft
- Used Resnet-50 as feature extractor for CNN Encoder
- NLTK used for pre-processing the captions.
- Word Embeddings are generated from the captions itself.
- Decoder is just a single layered RNN using LSTM cells.
- Trained for nearly 5 hours on GPU, still okayish accuracy.

Also, the Paper's folder contains papers read during the implementation of the project. 
Hope you read it too, for better understanding of the subject.
