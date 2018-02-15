# Semantic Segmentation project

Goals:
  * Implement a Fully Convolutional Network (FCN).
  * Use it to perform semantic segmentation on images, detecting which pixels belong to the road.

[//]: # (Image References)
[example-1]: ./images/example1.png "Example 1"
[example-2]: ./images/example2.png "Example 2"
[example-3]: ./images/example3.png "Example 3"


## Dependencies
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

## [Rubric](https://review.udacity.com/#!/rubrics/989/view) Points

### Build the Neural Network

#### Does the project load the pretrained vgg model?
It does, refer to `main.py` lines 20-42. In particular, we do:
```python
tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
graph = tf.get_default_graph()
```
And from the graph we obtain the pretrained tensors by name.

#### Does the project learn the correct features from the images?
It does, refer to `main.py` lines 46-63. It's leveraging the VGG model and adding the upsampling layers, following the model described in the [Fully Convolutional Networks for Semantic Segmentation](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf) whitepaper.

#### Does the project optimize the neural network?
It does, refer to `main.py` lines 67-81. We compute the softmax cross entropy between logits and labels and use an Adam algorithm optimizer to minimize the cross entrpy loss.

#### Does the project train the neural network?
Yes. Refer to `main.py` lines 85-109. It runs as per the specified number of epochs, using the batch_size parameter to obtain batched sets of training data. The loss of the network is printed while the network is training after each batch is processed.

### Neural Network Training

#### Does the project train the model correctly?
On average, the model decreases loss over time. The gains are big early on in the training but seem to stabilize by epoch 6.

#### Does the project use reasonable hyperparameters?
I found that the cross entropy loss doesn't decrease much after 5 or 6 epochs, as it appears to get stuck in some local minima with a loss value between 0.15 and 0.08. I ran up to 20 epochs but loss didn't move more. As seen in `main.py` lines 119-120 I left the code running only 8 epochs and the batch size to be 12. This was a good batch size given the memory constraints in my GPU.

#### Does the project correctly label the road?
The network does appear to correctly identify the road on all pictures with very little bleeding into non-road areas as seen in the following examples.

Example 1:

![Example 1][example-1]

Example 2:

![Example 2][example-2]

Example 3:

![Example 3][example-3]
