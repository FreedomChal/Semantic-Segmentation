#!/usr/bin/env python3
import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

# Hyperparameters
epochs = 64
batch_size = 32


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    vgg_tag = "vgg16"

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()
    w1 = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3 = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4 = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7 = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)


    return w1, keep, layer3, layer4, layer7
    
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    
    # This model is build generally according to the decoder described in the paper which goes with the vgg model, but with some changes.
    
    conv_1 = tf.layers.conv2d(vgg_layer7_out, 
                                21,
                                1, 
                                strides = (1, 1), 
                                padding="same", 
                                kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3),
                                name = "conv_1")

    conv_2 = tf.layers.conv2d(conv_1,
                                21,
                                2,
                                strides = (1, 1),
                                padding="same",
                                kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3),
                                name = "conv_2")

    conv_3 = tf.layers.conv2d(conv_2,
                                512,
                                4,
                                strides = (1, 1),
                                padding="same",
                                kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3),
                                name = "conv_3")

    conv1 = tf.layers.conv2d_transpose(conv_3, 
                                        256,
                                        4,
                                        strides = (2, 2),
                                        padding="same",
                                        kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3),
                                        name = "conv1")

    conv_4 = tf.layers.conv2d(conv1,
                                384,
                                1,
                                strides = (1, 1),
                                padding="same",
                                kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3),
                                name = "conv_5")


    conv_input2 = tf.layers.conv2d(vgg_layer4_out,
                                384,
                                1,
                                strides = (1, 1),
                                padding="same",
                                kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3),
                                name = "conv_input2")

    skip1 = tf.add(conv_4, conv_input2)

    conv3 = tf.layers.conv2d_transpose(skip1,
                                        384,
                                        4,
                                        strides = (2, 2),
                                        padding="same",
                                        kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3),
                                        name = "conv3")

    conv_input3 = tf.layers.conv2d(vgg_layer3_out,
                                384,
                                1,
                                strides = (1, 1),
                                padding="same",
                                kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3),
                                name = "conv_input3")


    skip2 = tf.add(conv3, conv_input3)
    
    output = tf.layers.conv2d_transpose(skip2, 
                                        num_classes,
                                        16,
                                        strides = (8, 8),
                                        padding="same", 
                                        kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3),
                                        name = "output_layer")
    return output
tests.test_layers(layers)


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
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = correct_label))
    
    # I got the code below from https://stackoverflow.com/questions/46615623/do-we-need-to-add-the-regularization-loss-into-the-total-loss-in-tensorflow-mode
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    reg_constant = 0.001
    loss = cross_entropy_loss + reg_constant * tf.reduce_sum(reg_losses)
    
    train_op = optimizer.minimize(loss)

    return logits, train_op, cross_entropy_loss
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image, correct_label, keep_prob, learning_rate, accuracy = None):
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
    :param accuracy: TF Operation to calculate the accuracy of the network
    """

    learning_rate_value = 0.001 # The learning rate value assigned to the learning rate placeholder.
    keep_prob_value = 0.5 # The keep prob value used for dropout.

    for epoch in range(epochs):
        print("Epoch " + str(epoch + 1))
        batch_number = 0 # Used for printing out which batch the training is on
        for image, label in get_batches_fn(batch_size):
            batch_number += 1
            if accuracy != None:
                op, loss, accuracy_numerical = sess.run([train_op, cross_entropy_loss, accuracy], feed_dict = {input_image: image, correct_label: label, learning_rate: learning_rate_value, keep_prob: keep_prob_value})
                print("Epoch " + str(epoch + 1) + " Batch " + str(batch_number) + " Loss: " + str(loss) + " Accuracy: " + str(accuracy_numerical))
            else: # If there is a problem calculating the accuracy, just print out the loss to prevent an error. This is most likely no longer necessary because the accuracy calculation error it was meant to fix was resolved.
                op, loss = sess.run([train_op, cross_entropy_loss], feed_dict = {input_image: image, correct_label: label, learning_rate: learning_rate_value, keep_prob: keep_prob_value})
                print("Epoch " + str(epoch + 1) + " Batch " + str(batch_number) + " Loss: " + str(loss))

    pass
tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)  # KITTI dataset uses 160x576 images
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        learning_rate = tf.placeholder(tf.float32) # Make a placeholder for learinig rate.
        
        input_image, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path) # Load the vgg model.
        layer_output = layers(layer3_out, layer4_out, layer7_out, num_classes) # Load the decoder part of the model.

        labels = tf.placeholder(shape = tuple([None]) + image_shape + tuple([num_classes]), name = "labels", dtype = tf.float32) # The placeholder for labels.

        logits, train_op, cross_entropy_loss = optimize(layer_output, labels, learning_rate, num_classes)

        correct_predictions = tf.cast(tf.equal(tf.argmax(labels, 3), tf.argmax(layer_output, 3)), tf.float32) # For each prediction, calculate whether or not the case the model thought was most likely is the true case.
        accuracy = tf.reduce_mean(correct_predictions) # Take the mean of the predictions the model correctly predicted to calculate the accuracy.

        sess.run(tf.initializers.global_variables())

        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image, labels, keep_prob, learning_rate, accuracy = accuracy)

        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
