#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/CNN_Car-Behavioral.png "Model Visualization"
[image2]: ./examples/center_2017_03_16_23_27_43_701.jpg "Center Lane Driving"
[image3]: ./examples/center_start.jpg "Recovery Image Start"
[image4]: ./examples/center_middle.jpg "Recovery Image Middle"
[image5]: ./examples/center_end.jpg "Recovery Image End"
[image6]: ./examples/center_original.jpg "Normal Image"
[image7]: ./examples/center_flipped.jpg "Flipped Image"
[image8]: ./examples/camera_center.jpg "Normal Center Camera Image"
[image9]: ./examples/camera_left.jpg "Left Camera Image"
[image10]: ./examples/camera_right.jpg "Right Camera Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.
The data_io.py file contains the code for data input including: loading, normalization, augmentation, and generation of training and validation data sets.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network inspired by the NVIDIA architecture described here, http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf.
It is an end-to-end model that have interspaced the layers in the NVIDIA design with Spatial Dropouts and a single Dropout (model.py lines 17-33)

The model includes ReLU and ELU (Exponential Linear Unit) layers to introduce nonlinearity (model.py line 20), and the data is normalized in the after loading using simple arithmetics in (data_io.py line 39). 

####2. Attempts to reduce overfitting in the model

The model interlaces spatial dropout layers in between each original NVIDIA layer order to reduce overfitting (model.py even lines 20-26) as well as a single dropout after the full-connected layers. 

The model was trained and validated on different data sets to ensure that the model was not overfitting (model.py lines 37-65 and data_io.py lines 12-24). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an Nadam optimizer, so the learning rate was not tuned manually (model.py line 25).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, horizontally flipping images and filtering out sequential large changes in driving angles to deduce jitter.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the one used by NVIDIA for Dave-2. I thought this model might be appropriate because several project notes (introduction as a more powerful network in the project prep material) and vetted recommendations by previous student's blog, https://carnd-forums.udacity.com/cq/viewquestion.action?id=26214464&questionTitle=behavioral-cloning-cheatsheet all recommend this as as starting point. 

In order to gauge how well the model was working, I split my image and steering angle data into a training (70%) and validation set (30%) using a variable VALIDAION_RATIO (data_io.py line 9). I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. Also in 10 epochs loss sporadically improved only 3-4 rounds (1-2 epochs at the beginning and at a random later epoch)  This implied that the model was overfitting.

To combat the overfitting, I modified the model so I tried adding dropout layers using eithr ReLU or ELU. ELU is theoretically  superior due to its solution to the Vanishing gradient problem . However, in practice I found ReLU to work well in the convolutional layers while fully-connected layers behaviorally seemed to learn faster with ELU.

The final model uses interlaced spatial dropout layers with ReLU in between each original convolutional NVIDIA layer. All spatial dropouts were set to be the same and found 0.2 to work well. To further address overfitting a Dropout layer of 0.5 was added prior to the fully-connected layers.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I created recovery data and augmented the data as explained in 3. Creation of the Training Set & Training Process.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes:
From input to end:
* 5x5 convolutional filter
* SpatialDropout
* 5x5 convolutional filter
* SpatialDropout
* 5x5 convolutional filter
* SpatialDropout
* 3x3 convolutional filter
* SpatialDropout
* 3x3 convolutional filter
* SpatialDropout
* 100 Fully-connected
* 50 Fully-connected
* 10 Fully-connected
* Dropout

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded three laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover after veering close to the edge of the lanes. These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

To augment the trainng data sat, I also flipped images and angles thinking that this would reduce the bias of the training data created with images circling the track in the same direction. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

To further augment the trainng data sat, I also used the left and right auxilliary camera images. 

For example, here is the center image along with it's left, right and their adjusted steering:

![alt text][image8]
![alt text][image9]
![alt text][image10]

In addition I also filtered out large changes in steering control.

After the collection process, I had X number of data points. I then preprocessed this data by simple arithmetics by "Normalization" (i.e. center around 0.5 and divide by 255)

After initial loading I randomly shuffled the data set and put 30% of the data into a validation set (data_io.py line 109 and 140) leaving 70% for training. 

The training data with augmentation was used to train the model. The validation, composed only of original images, set helped determine if the model was over or under fitting. The ideal number of epochs was 10 (suggested by the blog) as evidenced by inferior comparison of performance with a 25 epoch trained model. I used an Nadam optimizer so that manually training the learning rate wasn't necessary.
