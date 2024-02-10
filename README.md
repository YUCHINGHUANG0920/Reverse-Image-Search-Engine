
# Reverse Image Search Engine

## Overview
This project's purpose is to build a reverse image search engine using ResNet-152, Milvus, and Flask. We first use PyTorch to re-train a pre-trained ResNet-152 model with trainable parameters only at 2 layers. After that, we use the resulting ResNet-152 model to encode the Stanford Dogs Dataset into vectors. Then, we use Milvus to create a collection and insert the vectors in the collection. Finally, we build a website interface using Flask.  

When an input image comes in, our application first uses the resulting ResNet-152 model to translate the image into a vector. Since all the vectors in the collection are normalized, the query vector is then used by the cosine similarity algorithm to find the most related vectors in the collection through Milvus. The returned vectors then correspond of the images in the Stanford Dogs Dataset, and the images are returned as the most related results.

## Data Source
1. Input data for the ResNet-152 model (Download the file from the below link, unzip it and put it in the **deep_learning/input/model** folder):
   https://www.kaggle.com/code/halfendt/dog-breed-classifier-pytorch-resnet-152/output
2. Images for displaying (Download the file from the below link and unzip it. Then follow the path: archive/images, there is the **Images** folder. Select all files in the **Images** folder and put them in the **deep_learning/input/stanford-dogs-dataset** folder):
   https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset/data

&nbsp;

![Website Screeshot](https://github.com/KUANCHENGFU/Reverse-Image-Search-Engine/blob/main/static/screenshot.png)