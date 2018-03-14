import tensorflow as tf
import numpy as np
import os
from glob import glob
import random
import scipy.misc
import zipfile
import shutil
from urllib.request import urlretrieve
from tqdm import tqdm


class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num

def maybe_download_pretrained_vgg(data_dir):
    """
    Download and extract pretrained vgg model if it doesn't exist
    :param data_dir: Directory to download the model to
    """
    vgg_filename = 'vgg.zip'
    vgg_path = os.path.join(data_dir, 'vgg')
    vgg_files = [
        os.path.join(vgg_path, 'variables/variables.data-00000-of-00001'),
        os.path.join(vgg_path, 'variables/variables.index'),
        os.path.join(vgg_path, 'saved_model.pb')]

    missing_vgg_files = [vgg_file for vgg_file in vgg_files if not os.path.exists(vgg_file)]
    if missing_vgg_files:
        # Clean vgg dir
        if os.path.exists(vgg_path):
            shutil.rmtree(vgg_path)
        os.makedirs(vgg_path)

        # Download vgg
        print('Downloading pre-trained vgg model...')
        with DLProgress(unit='B', unit_scale=True, miniters=1) as pbar:
            urlretrieve(
                'https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip',
                os.path.join(vgg_path, vgg_filename),
                pbar.hook)

        # Extract vgg
        print('Extracting model...')
        zip_ref = zipfile.ZipFile(os.path.join(vgg_path, vgg_filename), 'r')
        zip_ref.extractall(data_dir)
        zip_ref.close()

        # Remove zip file to save space
        os.remove(os.path.join(vgg_path, vgg_filename))

def gen_batch_function(data_folder, image_shape):
    """
    Generate function to create batches of training data
    :param data_folder: Path to folder that contains all the datasets
    :param image_shape: Tuple - Shape of image
    :return:
    """
    def get_batches_fn(batch_size):
        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """
        image_paths = glob(os.path.join(data_folder, '*', '*.png'))

        label_classes = {0: [1,0,0,0], 1: [0,1,0,0], 2: [0,0,1,0], 4: [0,0,0,1]}

        random.shuffle(image_paths)
        for batch_i in range(0, len(image_paths), batch_size):
            images = []
            labels = []
            for image_file in image_paths[batch_i:batch_i+batch_size]:
                image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
                images.append(image)
                label_class = int(os.path.basename(os.path.dirname(image_file)))
                labels.append(label_classes[label_class])

            yield np.array(images), np.array(labels)
    return get_batches_fn

def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer7_out)
    """
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()
    vgg_input_tensor = graph.get_tensor_by_name(vgg_input_tensor_name)
    vgg_keep_prob_tensor = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    vgg_layer7_out_tensor = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return vgg_input_tensor, vgg_keep_prob_tensor, vgg_layer7_out_tensor

def layers(vgg_layer7_out, keep_prob, image_shape, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    mu = 0
    sigma = 0.1
    flat_layer7 = tf.contrib.layers.flatten(vgg_layer7_out)
    layer7_size = image_shape[0] * image_shape[1] * num_classes
    F_W = tf.Variable(tf.truncated_normal((layer7_size, 100), mean = mu, stddev = sigma, dtype=tf.float32))
    F_b = tf.Variable(tf.zeros(100))
    fully_connected = tf.matmul(flat_layer7, F_W) + F_b
    activated = tf.nn.relu(fully_connected)
    dropout = tf.nn.dropout(activated, keep_prob)
    F_W = tf.Variable(tf.truncated_normal((100, num_classes), mean = mu, stddev = sigma, dtype=tf.float32))
    F_b = tf.Variable(tf.zeros(num_classes))
    logits = tf.matmul(dropout, F_W) + F_b
    return logits

def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=correct_label, logits=logits)
    cross_entropy_loss = tf.reduce_mean(cross_entropy) + sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    optimizer = tf.train.AdamOptimizer(learning_rate)
    training_operation = optimizer.minimize(cross_entropy_loss)
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(correct_label, 1))
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return logits, training_operation, cross_entropy_loss, accuracy_operation

def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, accuracy_operation, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    print("Training...")
    print()
    sess.run(tf.global_variables_initializer())
    for epoch in range(epochs):
        print("EPOCH {} ...".format(epoch+1))
        for image, label in get_batches_fn(batch_size):
            sess.run(train_op, feed_dict={input_image: image, correct_label: label, keep_prob: 0.8, learning_rate: 1e-4})
            loss =  sess.run(cross_entropy_loss, feed_dict={input_image: image, correct_label: label, keep_prob: 0.8, learning_rate: 1e-4})
            print("Cross Entropy Loss = {:.3f}".format(loss))
            validation_accuracy = sess.run(accuracy_operation, feed_dict={input_image: image, correct_label: label, keep_prob: 0.8, learning_rate: 1e-4})
            print("Validation Accuracy = {:.3f}".format(validation_accuracy))
    print()

def run():
    epochs = 20
    batch_size = 10
    num_classes = 4
    image_shape = (160, 576)
    data_dir = './data'
    export_dir = './output'

    # Download pretrained vgg model
    maybe_download_pretrained_vgg(data_dir)

    builder = tf.saved_model.builder.SavedModelBuilder(export_dir)

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = gen_batch_function(os.path.join(data_dir, 'capture'), image_shape)


        # Build NN using load_vgg, layers, and optimize function
        input_image, keep_prob, layer7_out = load_vgg(sess, vgg_path)
        last_layer = layers(layer7_out, keep_prob, image_shape, num_classes)
        correct_label = tf.placeholder(tf.int32, shape=[None, num_classes], name="correct_label")
        learning_rate = tf.placeholder(tf.float32, name="learning_rate")
        logits, train_op, cross_entropy_loss, accuracy_operation = optimize(last_layer, correct_label, learning_rate, num_classes)

        # Train NN using the train_nn function
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, accuracy_operation, input_image, correct_label, keep_prob, learning_rate)

        builder.add_meta_graph_and_variables(sess, ["gauss-capstone-vgg16"])

    builder.save()

if __name__ == '__main__':
    run()
