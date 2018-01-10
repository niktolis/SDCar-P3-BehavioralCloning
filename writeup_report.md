# **Behavioral Cloning**

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./Figures/Try07.png "7th train"
[image2]: ./Figures/Try08.png "8th train"
[image3]: ./Figures/Try13.png "13th train"
[image4]: ./images/centerTrack1/center_2017_11_28_18_33_14_651.jpg "Center Track 1"
[image5]: ./images/centerTrack2/center_2017_11_30_18_23_41_656.jpg "Center Track 2"
[image6]: ./images/Retrieve1/center_2017_11_28_18_53_35_924.jpg "Retrieve Left 1"
[image7]: ./images/Retrieve1/center_2017_11_28_18_53_36_062.jpg "Retrieve Left 2"
[image8]: ./images/Retrieve1/center_2017_11_28_18_53_36_333.jpg "Retrieve Left 3"
[image9]: ./images/Retrieve2/center_2017_11_28_18_53_50_208.jpg "Retrieve Left 1"
[image10]: ./images/Retrieve2/center_2017_11_28_18_53_50_550.jpg "Retrieve Left 2"
[image11]: ./images/Retrieve2/center_2017_11_28_18_53_50_822.jpg "Retrieve Left 3"
[image12]: ./images/flipped/left_2017_11_28_18_33_14_651.jpg "Normal image"
[image13]: ./images/flipped/left_2017_11_28_18_33_14_651_flipped.jpg "Flipped image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* [model.py](./model.py) containing the script to create and train the model,
* [drive.py](./drive.py) for driving the car in autonomous mode,
* [model.h5](./model.h5) containing a trained convolution neural network,
* [writeup_report.md](./writeup_report) which is this file summarizing the results.

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64. Dense layers are included on the final stages that lead to the single output of the predicted steering angle (model.py lines 110-129).

The model includes RELU layers to introduce nonlinearity across all convolution and dense layers. Part of the preprocessing of the image is done in the generator, in specific the change of colorspace from RGB to YUV (model.py line 61). The rest of the preprocessing is done inside the model and it consists of:

* The cropping of the image (model.py line 112)
* The resize of the image (model.py line 113)
* The normalization of the image data (model.py line 115)


#### 2. Attempts to reduce overfitting in the model

The model needed only one dropout layer in order to reduce overfitting in between the change from convolution to dense layers (model.py line 123).

The model was trained and validated on different data sets with **75%** of the data used for training and the rest **25%** for validation. This was made to ensure that the model was not overfitting (code line 25). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track. Videos from both tracks were made and are provided:

* [Track1](./track1_video.mp4)
* [Track2](./track2_video.mp4)

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 136). The tuning was mostly focused on:
* The number of epochs which in the end was set to 7 cause greater numbers only led to overfitting of the model
* The percentage of the dropout layer
* The correction factor of the steering angle on left and right images

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road.

For the first track I used a combination of center lane driving, recovering from each side of the road and some specific smooth cornering with relatively slow speed.

For the second track I decided to drive not in the center but rather on the right lane which made the training a bit more difficult. I also added some recovery from the left lane as I noticed that if an accidental lane change was made the car was following then the left lane until the end.

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to follow the instructions of the project as it was shown on the lesson. The hint of using the NVIDIA proposed architecture proved to be a solid choice.

I also added the proposed from the authors cropping and resizing of the images as this greatly reduced the processing time on my rig. Also I took into account the differences of the lighting conditions which were even worse on second track and I changed the color space from RGB to YUV to reduce the effect the lighting conditions and change of tarmac have on the training of the model.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. Especially as I increased the dataset by adding the second track.

To combat the overfitting, I modified the model by adding a dropout layer between the convolutional and dense model. Below you can see a difference before and after the addition

![alt text][image1]![alt text][image2]

As I saw that both training and validation loss had a decreasing trend I decided to increase the number of epochs and see how low the validation loss can go. I noticed that after epoch no 7 there was no point on keep training as the results didn't get any better. The last training is with 7 epochs is shown below:

![alt text][image3]

At the end of the process, the vehicle is able to drive autonomously around the first track without leaving the road but also around the second track without leaving its correct lane.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes:

| Layer                  |  Description                   |
|:----------------------:|:------------------------------:|
| Convolution 5x5        | stride 2x2, outputs  98x31x24  |
| RELU                   |                                |
| Convolution 5x5        | stride 2x2, outputs  47x14x36  |
| RELU                   |                                |
| Convolution 5x5        | stride 2x2, outputs  22x5x48   |
| RELU                   |                                |
| Convolution 3x3        | stride 1x1, outputs  20x3x64   |
| RELU                   |                                |
| Convolution 3x3        | stride 1x1, outputs  18x1x64   |
| RELU                   |                                |
| Dropout                |                                |
| Fully Connected 1      | outputs 100                    |
| RELU                   |                                |
| Fully Connected 2      | outputs 50                     |
| RELU                   |                                |
| Fully Connected 3      | outputs 10                     |
| RELU                   |                                |
| Output                 |                                |


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving for 3 runs. One with fast speed but a little bit more careless on the corners one slower with using brake on corners for smoother steering and one run only recording the corners. Here is an example image of center lane driving from the first track:

![alt text][image4]

Same approach I used also for the second track. I did 3 runs at first but I was driving on the center of the road. In order to make the project a bit more challenging a later replaced this data with driving only on the right lane. This gave an extra problem to try and recover if there was a sudden change of lanes.

From the second track driving on the center of right lane:

![alt text][image5]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover from curves. These images show what a recovery looks like starting from a left curve:

![alt text][image6]
![alt text][image7]
![alt text][image8]

And from a right curve:

![alt text][image9]
![alt text][image10]
![alt text][image11]

Then I repeated this process on track two in order to get more data points I also had some recovery runs from the left back to the right lane.

To augment the dataset, I decided to flip all the left and right images included on the training dataset since I was driving the tracks only one direction and I wanted to remove the bias on the model depending on the direction of the track, especially the first one which has only clockwise turns. Here is an example of flipping.

![alt text][image12]
![alt text][image13]

During flipping I also get the opposite of the steering measurement. I use as additional input on training the left and right images applying a correction which was tuned during the training process.

After the collection process, I had **16820** number of data points before the augmentation. After they were shuffled they were split to:

* **75%** on training dataset resulting in 12615 data points.
* **25%** on validation dataset resulting in 4205 data points.

After that the training generator was implemented in order to augment the data by taking into account the left and right images and also flipping them and applying the opposite of the measurement. In each batch iteration the batches were reshuffled so the model is trained without getting biased by the sequence of images.

A few things needed extra care as the model worked well from the start especially for the first track. To tackle some problems I decided to switch colorspace to YUV and this helped the model on the simulation to pass the difficult parts e.g. the stone bridge and the corners near the lake. After it successfully finished the first track I proceeded to the second track which also was completed. Lastly I tried to increase the speed of the car and also I tuned it to be a little smoother by adding a low pass filter on the speed controller in drive.py (lines 26-55).
