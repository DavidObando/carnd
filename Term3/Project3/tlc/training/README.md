[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)
This repository contains a SSD model made by reference to [cory8249](https://github.com/cory8249/ssd_keras.git).

# SSD for Traffic Light Detection

This is 'Traffic Light Detection' for System Integraton Project in Udacity 'Self-Driving Car Engineer Nanodegree Program'.

For more details, please refer to 

- [SSD: Single Shot MultiBox Detector](https://github.com/weiliu89/caffe/tree/ssd)
- [arXiv paper](http://arxiv.org/abs/1512.02325).

This code was tested with 

    Keras==2.0.8
    Pillow==2.2.1
    h5py==2.6.0
    numpy==1.13.1
    protobuf==3.5.2
    scipy==0.19.1
    tensorflow==1.3.0

# training images from bag file

    python bag_to_images.py bag/just_traffic_light.bag ssd/images /current_pose

# training images from simulator

You can also download the latest data used to train the model (captured from the simulator) here: [capture-2.zip](https://1drv.ms/u/s!AtMG4jW974a6m8B-Q0A3tc2JbCigPw).

# annotating data

I used to input bounding boxes with opencv_annotation

    opencv_annotation -i=/to/images/folder -a=annotation.txt

then added classno every lines like this:

    frame000000.png 2 1 642 409 27 68
    ...

formated as 'image-file-basename classno  1 boundingbox-xmin ymin width height'
'1' means this line has one rectangle (= 4 integers)

Then convert the text file to pickle format version 2 for python2.7 required by the project.

    python gt_format.capture.py

see gt_format.py and edit your input/output file.


# training

copy your pickle file into ssd folder and train.

    cd ssd

edit SSD_train.py line 40 with your pickle file.
If need, you also need to edit your image folder at line 241.

run a training script

	python SSD_train.capture.py

you can get new checkout files in checkpoints/ folder.
