[image1]: https://github.com/FreedomChal/Semantic-Segmentation/blob/master/model_prediction_image_good_1.png "Model classification image  1"



# Semantic-Segmentation
Classwork for the Udacity Self Driving Car Nanodegree.

## Project Details

This repository contains a implementation of a Fully Convolutional Network (FCN), which is trained to classify each pixel of an image as "Road" or "Non-Road", which it can do effectively.

![alt text][image1]

### Files

* main.py is the file which contains the Model, Optimizer, and Training Pipeline. Running it trains the model.

* helper.py contains helper functions used by main.py.

* project_tests.py contains functions used for testing the model in main.py.

* runs/1552999337.6463115 contains images the Model was tested on. In these images, the pixels the Model classified as "Road" are shaded green.

## The Model

The Model is a Fully Convolutional Network (FCN) based off of [this paper](https://arxiv.org/pdf/1605.06211.pdf). It uses a pre-trained encoder, with a decoder slightly different than the one described in the paper, but with the same central features.
