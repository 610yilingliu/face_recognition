# Face Recognition for Recognizing Neighbor and Stranger

### Author: Yiling Liu (Individual Project)
### Student ID: 22214014

# Background Description

## Why Choose this Project

I am living in an apartment which has 7 floors. The neighbor who living in 7th floor is a kind, talkative man  - we call him uncle Fu. He always knock my door and come to chat with my father. But the security of our apartment is quite lax, every few days there will be some salesman knock our door to sell their products. It is quite annoying, and it is not polite to open the door and drive them away directly. There are two ways to solve this problem: to spend about 2,000 CNY to buy a new door with sight window, or to recognize who is knocking the door without opening the interior door (with ai tech and Raspberry Pi. Taking photo while the body sensor module detect there is a person at the door, and analyze if that photo contains uncle Fu or not) so we can pretend there is nobody at home and the salesman will leave.

This project will not spend too much time in training model, too. Unlike the large datasets on Kaggle like recognizing the right whales, it is impossible for me to take hundreds of photo of uncle Fu so the training data will not be sufficient.

## Techs will be used in this Project(Initial plan)

- CNN - Feature detection
- Siamese Network
- OpenCV - Locate human face from a photo


## Techs used in this Project(Final version)

- CNN
- Siamese Network
- nary-tree - Manage file system to find pictures in the appointed directory
- OpenCV - Crop and resize image
- dlib - cv2.CascadeClassifier is not precise enough, dlib is better

## Existed Face Recognition Libraries
- [deepface](https://github.com/serengil/deepface)
- [face recognition](https://github.com/ageitgey/face_recognition)
-  ......

Although those libraries on Github is robust and mature, I still need to train my own neural network to study how to do it without an all-in library.

It is quite time consuming for me to train a model to label and crop the human face region from the whole picture, so I use dlib directly to do this work.

# Project Scope and Limitations

## Project Scope

This project just do the coding part (generate a model and recognize face in the imported photo).

Other steps, including use Raspberry Pie to take photo and transmit it to cloud server for computer inside room to download are related to IoT and Cloud Computing but not AI. So they could be ignored temporarily in this AI unit project.


## What to be Improved

To make it easier for project presentation and the marking of teacher, I did a lots of I/O and save some middleware(images, csv files) inside the disk.

The improvements we can make in this project are:

1. For `build_model`, do not save cropped images directly in a disk, save their numpy array as a `.npy` file instead of cropped image + file path in a `.csv` file
   
2. For `detect_face`, valid face could be saved in a `.npy` file and load while using it, instead of read all images and convert it into numpy array each time while running the program.

3. Noises could be add in the training pictures (blur, add some rare color block, .etc) to recognize photos in complicated environments.

# Training process (directory `build_model`)

The training data I use is [AT&T Database of Faces](https://www.kaggle.com/kasikrit/att-database-of-faces) from Kaggle which contains 40 people's face, these 40 people could be previewed in the following figure

<center><img src="pics/face_preview.png"  alt="Original Image", align = "center"></center>

*<center>Fig1. AT&T Dataset Preview</center>*

To meet the face cropped by dlib, I cropped the faces in this dataset - remove hair from the photo

<center><img src="pics/original.png"  alt="Original Image", align = "center"></center>

*<center>Fig2. Original Image</center>*

<center> <img src="pics/cropped.png"  alt="Cropped Image"></center>

*<center>Fig3. Cropped Image</center>*

I sampled 30,000 pairs of data from training images to build the dataset (80% training set, 20% test set) to train the for 25 epochs.

The loss of each epoch are shown in the following figure

<center><img src="pics/epochs_losses.png"  alt="loss in each epoch" align=center /></center>

*<center>loss in each epoch</center>*


The 'Reward' between two adjacent epochs could be defined as

$$
\frac{x_{i - 1} - x_i}{x_{i - 1}}
$$

It is also known as `f'(x)`  of loss in each epoch

<center><img src="pics/epochs_rewards.png"  alt="loss in each epoch" align=center /></center>

The highest reward happen between epoch one and two and get fluctuated in the following epochs. It is clear that the losses are decreasing continuously during the training process through