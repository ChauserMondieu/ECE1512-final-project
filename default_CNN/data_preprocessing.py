import tensorflow as tf
import numpy as np
from six.moves import xrange
import os

# set up the default parameters
image_size = 24
# Global constants describing the CIFAR-10 data set.
num_classes = 10
num_example_for_train = 50000
num_example_for_evaluation = 10000


# main function
# Returns an object representing a single example, with the following fields:
# height: number of rows in the result (32)
# width: number of columns in the result (32)
# depth: number of color channels in the result (3)
# key: a scalar string Tensor describing the filename & record number for this example.
# label: an int32 Tensor with the label in the range 0..9.
# uint8image: a [height, width, depth] uint8 Tensor with the image data
def read_cifar10(filename_queue):
    # define a structure object which could contain different attributes
    class CIFAR10Record( object ):
        pass

    # result is the passing parameter in this function
    result = CIFAR10Record()

    # Dimensions of the images in the CIFAR-10 dataset.
    label_bytes = 1
    result.height = 32
    result.width = 32
    result.depth = 3
    image_bytes = result.height * result.width * result.depth
    # Every record consists of a label followed by the image, with a
    # fixed number of bytes for each.
    record_bytes = label_bytes + image_bytes

    ''' define the file reading rules: reader'''
    # Read a record, getting file_names from the filename_queue.  No
    # header or footer in the CIFAR-10 format, so we leave header_bytes
    # and footer_bytes at their default of 0.
    reader = tf.FixedLengthRecordReader( record_bytes=record_bytes, name='data_reader' )
    ''' start reading the file'''
    # read the file name as well as the value in the file into a list
    # format key: file_name, value: [length = record_bytes]
    result.key, value = reader.read( filename_queue )

    # Convert from a string to a vector of uint8 that is record_bytes long
    # doesn't change the shape of vector but change the data type
    # different from the function: tf.cast.
    record_bytes = tf.decode_raw( value, tf.uint8 )

    # The first bytes represent the label, which we convert from uint8->int32.
    # tf.strided_slice: choose data of index from 0 to label_bytes in the record_bytes
    result.label = tf.cast(
        tf.strided_slice( record_bytes, [0], [label_bytes] ), tf.int32, name='label_get' )

    # The remaining bytes after the label represent the image, which we reshape
    # from [depth * height * width] to [depth, height, width].
    depth_major = tf.reshape(
        tf.strided_slice( record_bytes, [label_bytes],
                          [label_bytes + image_bytes] ),
        [result.depth, result.height, result.width], name='image_get' )
    # Convert from [depth, height, width] to [height, width, depth].
    result.uint8image = tf.transpose( depth_major, [1, 2, 0] )

    return result


# defining the output form of the preprocessed images
# Returns:
# images: Images. 4D tensor of [batch_size, height, width, 3] size.
# labels: Labels. 1D tensor of [batch_size] size.
def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
    # Create a queue that shuffles the examples, and then
    # read 'batch_size' images + labels from the example queue.
    num_preprocess_threads = 16
    # make sure that each out queue function only takes up to three-batch-size data
    if shuffle:
        images, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples,
            name='input_feedin_shuffle')
    else:
        images, label_batch = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            name='input_feedin')

    '''
    # store the corresponding graph data into tensor board
    sess = tf.Session()
    writer = tf.summary.FileWriter( "logs/" )

    # Display the training images in the visualizer.
    distorted_images = tf.summary.image( 'images', images, max_outputs=10 )
    # write the corresponding images into tensor board
    image_summary = sess.run(distorted_images)
    writer.add_summary(image_summary)
    # close the current session
    sess.close()
    writer.close()
    '''
    # Transfer label into 1D array
    labels = tf.reshape( label_batch, [batch_size] )

    return images, labels


# generating the training batches
def crop_distorted_inputs(data_dir, batch_size):
    ''' prepare the names of input files '''
    # import file names into file_names variable
    # format: [name1, name3, name3, name4, name5, name6]
    file_names = [os.path.join( data_dir, 'data_batch_%d.bin' % i )
                  for i in xrange( 1, 6 )]
    # if file not find, throw exceptions
    for f in file_names:
        if not tf.gfile.Exists( f ):
            raise ValueError( 'Failed to find file: ' + f )

    # Create a queue that produces the file_names to read.
    filename_queue = tf.train.string_input_producer( file_names )

    with tf.name_scope( 'train_input_data_augmentation' ):
        # Read examples from files in the filename queue.
        # only care for 2 things: read_input.uint8image & read_input.label
        read_input = read_cifar10( filename_queue )

        # all the process of pre-processing read_input.uint8image
        reshaped_image = tf.cast( read_input.uint8image, tf.float32, name='image_reshape' )

        height = image_size
        width = image_size

        # Image processing for training the network. Note the many random
        # distortions applied to the image.

        # Randomly crop a [height, width] section of the image.
        distorted_image = tf.random_crop( reshaped_image, [height, width, 3], name='image_crop' )

        # Randomly flip the image horizontally.
        distorted_image = tf.image.random_flip_left_right( distorted_image )

        # Normalize the brightness and contrast
        distorted_image = tf.image.random_brightness( distorted_image,
                                                      max_delta=63 )
        distorted_image = tf.image.random_contrast( distorted_image,
                                                    lower=0.2, upper=1.8 )

        # Subtract off the mean and divide by the variance of the pixels.
        float_image = tf.image.per_image_standardization( distorted_image )

        # Set the shapes of tensors.
        float_image.set_shape( [height, width, 3] )
        read_input.label.set_shape( [1] )

        # Ensure that the random shuffling has good mixing properties.
        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples = int( num_example_for_train *
                                  min_fraction_of_examples_in_queue )
        print( 'Filling queue with %d CIFAR images before starting to train. '
               'This will take a few minutes.' % min_queue_examples )

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch( float_image, read_input.label,
                                            min_queue_examples, batch_size,
                                            shuffle=True )


# generating the training batches
def crop_only_inputs(data_dir, batch_size):
    ''' prepare the names of input files '''
    # import file names into file_names variable
    # format: [name1, name3, name3, name4, name5, name6]
    file_names = [os.path.join( data_dir, 'data_batch_%d.bin' % i )
                  for i in xrange( 1, 6 )]
    # if file not find, throw exceptions
    for f in file_names:
        if not tf.gfile.Exists( f ):
            raise ValueError( 'Failed to find file: ' + f )

    # Create a queue that produces the file_names to read.
    filename_queue = tf.train.string_input_producer( file_names )

    with tf.name_scope( 'data_augmentation' ):
        # Read examples from files in the filename queue.
        # only care for 2 things: read_input.uint8image & read_input.label
        read_input = read_cifar10( filename_queue )

        # all the process of pre-processing read_input.uint8image
        reshaped_image = tf.cast( read_input.uint8image, tf.float32 )

        height = image_size
        width = image_size

        # Image processing for training the network. Note the many random
        # distortions applied to the image.

        # Randomly crop a [height, width] section of the image.
        cropped_image = tf.random_crop( reshaped_image, [height, width, 3] )

        # Subtract off the mean and divide by the variance of the pixels.
        float_image = tf.image.per_image_standardization( cropped_image )

        # Set the shapes of tensors.
        float_image.set_shape( [height, width, 3] )
        read_input.label.set_shape( [1] )

        # Ensure that the random shuffling has good mixing properties.
        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples = int( num_example_for_train *
                                  min_fraction_of_examples_in_queue )
        print( 'Filling queue with %d CIFAR images before starting to train. '
               'This will take a few minutes.' % min_queue_examples )

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch( float_image, read_input.label,
                                            min_queue_examples, batch_size,
                                            shuffle=True )


# generating the evaluation batches
def inputs(eval_data, data_dir, batch_size):
    if not eval_data:
        filenames = [os.path.join( data_dir, 'data_batch_%d.bin' % i )
                     for i in xrange( 1, 6 )]
        num_examples_per_epoch = num_example_for_train
    else:
        filenames = [os.path.join( data_dir, 'test_batch.bin' )]
        num_examples_per_epoch = num_example_for_evaluation

    for f in filenames:
        if not tf.gfile.Exists( f ):
            raise ValueError( 'Failed to find file: ' + f )

    # Create a queue that produces the filenames to read.
    filename_queue = tf.train.string_input_producer( filenames )
    with tf.name_scope( 'test_input' ):
        # Read examples from files in the filename queue.
        # only care for 2 things: read_input.uint8image & read_input.label
        read_input = read_cifar10( filename_queue )

        # all the process of pre-processing read_input.uint8image
        reshaped_image = tf.cast( read_input.uint8image, tf.float32 )

        height = image_size
        width = image_size

        # Image processing for evaluation.
        # Crop the central [height, width] of the image.
        resized_image = tf.image.resize_image_with_crop_or_pad( reshaped_image,
                                                                height, width )

        # Subtract off the mean and divide by the variance of the pixels.
        float_image = tf.image.per_image_standardization( resized_image )

        # Set the shapes of tensors.
        float_image.set_shape( [height, width, 3] )
        read_input.label.set_shape( [1] )

        # Ensure that the random shuffling has good mixing properties.
        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples = int( num_examples_per_epoch *
                                  min_fraction_of_examples_in_queue )

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch( float_image, read_input.label,
                                            min_queue_examples, batch_size,
                                            shuffle=False )
