[image1]: https://github.com/FreedomChal/Semantic-Segmentation/blob/master/model_prediction_image_good_1.png "Model classification image  1"
[image2]: https://github.com/FreedomChal/Semantic-Segmentation/blob/master/SemanticSegmentationModelVisualization.png "Model Visualization"
[image3]: https://github.com/FreedomChal/Semantic-Segmentation/blob/master/model_prediction_image_good_2.png "Model classification image  2"
[image4]: https://github.com/FreedomChal/Semantic-Segmentation/blob/master/model_prediction_image_good_3.png "Model classification image  3"
[image5]: https://github.com/FreedomChal/Semantic-Segmentation/blob/master/model_prediction_image_good_4.png "Model classification image  4"
[image6]: https://github.com/FreedomChal/Semantic-Segmentation/blob/master/model_prediction_image_bad_1.png "Bad model classification image 1"
[image7]: https://github.com/FreedomChal/Semantic-Segmentation/blob/master/model_prediction_image_bad_2.png "Bad model classification image 2"



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

The Model is a Fully Convolutional Network (FCN) based off of [this paper](https://arxiv.org/pdf/1605.06211.pdf). It uses a pre-trained encoder, with a decoder slightly different than the one described in the paper, but with the same central features, as shown below:

![alt text][image2]

This model is able to consistently predict road pixels with 96% accuracy, usually cleanly and effectively locating the road.

![alt text][image3]

![alt text][image4]

## Training

Early on, I used a batch size of 32. This value worked plenty well, so I stuck with it.
I trained the model for 64 epochs during the final training. I had tried 16 epochs previously, but the model did not do nearly as well when I did this. When I only ran training for 16 epochs, the model usually only got 92% training accuracy at best, verses the final consistent 96% accuracy. Also, the test images revealed multiple problems, most notably, trouble dealing with shadows. Increased epochs seemed to fix these problems.
After 64 epochs, the model was not training nearly as fast. Training for more epochs would likely result in an improvement in accuracy, but I chose not to because it would be time-consuming, and the model was already doing quite well.

![alt text][image5]

## Weaknesses/Improvements

![alt text][image6]

One somewhat strange weakness of the model is that it often classifies areas of car windshields as road. This may be due to the dark gray, road-like appearance of windshields from some angles. This does not happen in all images with car windshields in them, but it is not an uncommon occurence. This problem could likely be fixed by just training for more epochs.

Another weakness of the model is its tendency to classify areas near the road that look like road as road, as shown below.

![alt text][image7]

As with the car windshield problem, this weakness probably could be fixed or at least reduced by training for longer.

Another possible improvement to the model would be training data augmentation. By generating modified training data, the model could both train on more data, and get more unusual data, training the model to deal with dark, obstructed, or otherwise unusual images. I considered doing data augmentation, but I eventually chose not to.
