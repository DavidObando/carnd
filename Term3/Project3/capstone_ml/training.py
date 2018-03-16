import tensorflow as tf
import numpy as np
import os
from glob import glob
import random
import scipy.misc
import zipfile
import shutil
from urllib.request import urlretrieve


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
                image = (image - 128.) / 128. # normalize the image data
                images.append(image)
                label_class = int(os.path.basename(os.path.dirname(image_file)))
                labels.append(label_classes[label_class])

            yield np.array(images), np.array(labels)
    return get_batches_fn

def layers(input_layer, keep_probability, image_shape, num_classes):
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1

    # Layer 1: Convolutional. Input = image_shape[0] x image_shape[1] x 3. Output = 28 x 28 x 6.
    # new_height = (input_height - filter_height + 2 * P)/S + 1
    # 28 = ((image_shape[0] - fh)/S) + 1
    #(27 * S) + fh = image_shape[0]
    # S = 1, fh = image_shape[0] - 27
    F_W = tf.Variable(tf.truncated_normal((image_shape[0] - 27, image_shape[1] - 27, 3, 6), mean = mu, stddev = sigma, dtype=tf.float32))
    F_b = tf.Variable(tf.zeros(6))
    strides = [1, 1, 1, 1]
    padding = 'VALID'
    layer1 = tf.nn.conv2d(input_layer, F_W, strides, padding) + F_b

    # Activation 1.
    layer1 = tf.nn.relu(layer1)

    # Pooling 1. Input = 28x28x6. Output = 14x14x6.
    # new_height = (input_height - filter_height)/S + 1
    # 14 = ((28 - fh)/S) + 1
    # (13 * S) + fh = 28
    # S = 2, fh = 2
    ksize=[1, 2, 2, 1]
    strides=[1, 2, 2, 1]
    padding = 'VALID'
    pooling_layer1 = tf.nn.max_pool(layer1, ksize, strides, padding)
    
    # Dropout 1.
    pooling_layer1 = tf.nn.dropout(pooling_layer1, keep_probability)
    
    # Flatten 2a. Input = 14x14x6. Output = 1176.
    flatten_layer2a = tf.contrib.layers.flatten(pooling_layer1)

    # Layer 2b: Convolutional. Input = 14x14x6. Output = 10x10x16.
    # new_height = (input_height - filter_height + 2 * P)/S + 1
    # 10 = ((14 - fh)/S) + 1
    # (9 * S) + fh = 14
    # S = 1, fh = 5
    F_W = tf.Variable(tf.truncated_normal((5, 5, 6, 16), mean = mu, stddev = sigma, dtype=tf.float32))
    F_b = tf.Variable(tf.zeros(16))
    strides = [1, 1, 1, 1]
    padding = 'VALID'
    layer2b = tf.nn.conv2d(pooling_layer1, F_W, strides, padding) + F_b
    
    # Activation 2b.
    layer2b = tf.nn.relu(layer2b)

    # Pooling 2b. Input = 10x10x16. Output = 5x5x16.
    # new_height = (input_height - filter_height)/S + 1
    # 5 = ((10 - fh)/S) + 1
    # (4 * S) + fh = 10
    # S = 2, fh = 2
    ksize=[1, 2, 2, 1]
    strides=[1, 2, 2, 1]
    padding = 'VALID'
    pooling_layer2b = tf.nn.max_pool(layer2b, ksize, strides, padding)

    # Dropout 2b.
    pooling_layer2b = tf.nn.dropout(pooling_layer2b, keep_probability)

    # Flatten 2b. Input = 5x5x16. Output = 400.
    flatten_layer2b = tf.contrib.layers.flatten(pooling_layer2b)
    
    # Layer 2c: Convolutional. Input = 5x5x16. Output = 3x3x32.
    # new_height = (input_height - filter_height + 2 * P)/S + 1
    # 3 = ((5 - fh)/S) + 1
    # (2 * S) + fh = 5
    # S = 2, fh = 1
    F_W = tf.Variable(tf.truncated_normal((1, 1, 16, 32), mean = mu, stddev = sigma, dtype=tf.float32))
    F_b = tf.Variable(tf.zeros(32))
    strides = [1, 2, 2, 1]
    padding = 'VALID'
    layer2c = tf.nn.conv2d(pooling_layer2b, F_W, strides, padding) + F_b
    
    # Activation 2c.
    layer2c = tf.nn.relu(layer2c)

    # Pooling 2c. Input = 3x3x32. Output = 2x2x32.
    # new_height = (input_height - filter_height)/S + 1
    # 2 = ((3 - fh)/S) + 1
    # (1 * S) + fh = 3
    # S = 2, fh = 1
    ksize=[1, 1, 1, 1]
    strides=[1, 2, 2, 1]
    padding = 'VALID'
    pooling_layer2c = tf.nn.max_pool(layer2c, ksize, strides, padding)

    # Dropout 2c.
    pooling_layer2c = tf.nn.dropout(pooling_layer2c, keep_probability)

    # Flatten 2c. Input = 2x2x32. Output = 128.
    flatten_layer2c = tf.contrib.layers.flatten(pooling_layer2c)
    
    # Concat layers 2a, 2b, 2c. Input = 1176 + 400 + 128. Output = 1704.
    flat_layer2 = tf.concat([tf.concat([flatten_layer2b, flatten_layer2a], 1), flatten_layer2c], 1)
    
    # Layer 3: Fully Connected. Input = 1704. Output = 120.
    F_W = tf.Variable(tf.truncated_normal((1704, 120), mean = mu, stddev = sigma, dtype=tf.float32))
    F_b = tf.Variable(tf.zeros(120))
    fully_connected = tf.matmul(flat_layer2, F_W) + F_b
    
    # Activation 3.
    fully_connected = tf.nn.relu(fully_connected)
    
    # Dropout 3.
    fully_connected = tf.nn.dropout(fully_connected, keep_probability)

    # Layer 4: Fully Connected. Input = 120. Output = 84.
    F_W = tf.Variable(tf.truncated_normal((120, 84), mean = mu, stddev = sigma, dtype=tf.float32))
    F_b = tf.Variable(tf.zeros(84))
    fully_connected = tf.matmul(fully_connected, F_W) + F_b
    
    # Activation 4.
    fully_connected = tf.nn.relu(fully_connected)

    # Dropout 4.
    fully_connected = tf.nn.dropout(fully_connected, keep_probability)

    # Layer 5: Fully Connected. Input = 84. Output = num_classes.
    F_W = tf.Variable(tf.truncated_normal((84, num_classes), mean = mu, stddev = sigma, dtype=tf.float32))
    F_b = tf.Variable(tf.zeros(num_classes))
    logits = tf.matmul(fully_connected, F_W) + F_b
    
    # Dropout 5.
    logits = tf.nn.dropout(logits, keep_probability)

    return logits

def optimize(logits, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param logits: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
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
            validation_accuracy = sess.run(accuracy_operation, feed_dict={input_image: image, correct_label: label, keep_prob: 0.8, learning_rate: 1e-4})
            print("E{} - Cross Entropy Loss = {:.3f} - Validation Accuracy = {:.3f}".format(epoch+1, loss, validation_accuracy))
    print()

def run():
    epochs = 100
    batch_size = 400
    num_classes = 4
    image_shape = (300, 400)
    data_dir = './data'
    export_dir = './output'

    builder = tf.saved_model.builder.SavedModelBuilder(export_dir)

    with tf.Session() as sess:
        # Create function to get batches
        get_batches_fn = gen_batch_function(os.path.join(data_dir, 'capture'), image_shape)


        # Build NN using layers, and optimize function
        input_layer = tf.placeholder(tf.float32, (None, image_shape[0], image_shape[1], 3),  name="input_layer")
        keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        last_layer = layers(input_layer, keep_prob, image_shape, num_classes)
        correct_label = tf.placeholder(tf.int32, shape=[None, num_classes], name="correct_label")
        learning_rate = tf.placeholder(tf.float32, name="learning_rate")
        logits, train_op, cross_entropy_loss, accuracy_operation = optimize(last_layer, correct_label, learning_rate, num_classes)

        # Train NN using the train_nn function
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, accuracy_operation, input_layer, correct_label, keep_prob, learning_rate)

        builder.add_meta_graph_and_variables(sess, ["gauss-capstone"])

    builder.save()

if __name__ == '__main__':
    run()
