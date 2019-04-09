import tensorflow as tf
import numpy as np
import time
import math
import preprocessing as pr
import result_plot as pl
import visualization as vi


# set up input information
max_steps = 5000
batch_size = 128
data_dir = r'E:\WORK-PYTHON\tensor_test\cifar10_data\cifar-10-batches-bin'


# initialize weight variable，parameter wl control the degree of L2 normalization
# shape: [width, height, channel0, channel1]
# shape
def variable_with_weight_loss(shape, stddev, wl, name=None):
    var = tf.Variable( tf.truncated_normal( shape, stddev=stddev ), name=name )
    if wl is not None:
        weight_loss = tf.multiply( tf.nn.l2_loss( var ), wl, name='weight_loss' )
        tf.add_to_collection( 'losses', weight_loss )
        '''
        [tf.add_to_collection: function: add variable into a storing list]
        v1 = tf.get_variable(name='v1', shape=[1], initializer=tf.constant_initializer(0))
        tf.add_to_collection('loss', v1)
        print tf.get_collection('loss')
        <tensorflow.python.ops.variables.Variable object at 0x7f6b5d700c50>
        '''
    return var


'''data preparation session '''
# define the pre-processed training image & corresponding label
images_train, labels_train = pr.train_inputs( data_dir=data_dir,
                                              batch_size=batch_size )
# define the pre-scaled testing image & corresponding label
images_test, labels_test = pr.test_inputs( eval_data=True,
                                           data_dir=data_dir,
                                           batch_size=batch_size )

# define placeholder
# the first parameter type in here is batch_size rather than None
with tf.name_scope( 'placeholder' ):
    image_holder = tf.placeholder( tf.float32, [batch_size, 32, 32, 3], name='image_holder' )
    label_holder = tf.placeholder( tf.int32, [batch_size], name='label_holder' )

''' CNN construction session '''
# [1] first convolution layer
with tf.name_scope( '1_1st_convolution_layer' ):
    # parameter f=5 c0=3 s=1 c1=64
    # padding = same means that output weight = height = input/stride
    weight1 = variable_with_weight_loss( [5, 5, 3, 64], stddev=5e-2, wl=0.0, name='weight1' )  # 0.05
    vi.variable_summaries( weight1 )
    bias1 = tf.Variable( tf.constant( 0.0, shape=[64] ), name='bais1' )
    vi.variable_summaries( bias1 )

    kernel1 = tf.nn.conv2d( image_holder, weight1, strides=[1, 1, 1, 1],
                            padding='SAME' )
    conv1 = tf.nn.relu( tf.nn.bias_add( kernel1, bias1 ), name='conv1_output' )

# [2] first max pooling layer
with tf.name_scope( '2_1st_pooling_layer' ):
    pool1 = tf.nn.max_pool( conv1, ksize=[1, 3, 3, 1],
                            strides=[1, 2, 2, 1], padding='SAME',
                            name='pool1_output' )
    norm1 = tf.nn.lrn( pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1' )

# [3] second convolution layer
with tf.name_scope( '3_2nd_convolution_layer' ):
    # parameter f=5 c0=64 s=1 c1 = 64
    # padding = same
    weight2 = variable_with_weight_loss( [5, 5, 64, 64], stddev=5e-2, wl=0.0, name='weight2' )
    vi.variable_summaries( weight2 )
    bias2 = tf.Variable( tf.constant( 0.1, shape=[64] ), name='bias2' )
    vi.variable_summaries( bias2 )

    kernel2 = tf.nn.conv2d( norm1, weight2, strides=[1, 1, 1, 1],
                            padding='SAME' )
    conv2 = tf.nn.relu( tf.nn.bias_add( kernel2, bias2 ), name='Relu2_output' )

# [4] second max pooling layer
with tf.name_scope( '4_2nd_pooling_layer' ):
    norm2 = tf.nn.lrn( conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2' )
    pool2 = tf.nn.max_pool( norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                            padding='SAME', name='pool2_output' )

# [5] first fully connected layer
with tf.name_scope( '5_1st_Fully_Connected_layer' ):
    # input layer: [batch_size, 24, 24, 64] hidden layer: 384
    # reshape each tensor to 2d
    reshape = tf.reshape( pool2, [batch_size, -1] )
    dim = reshape.get_shape()[1].value

    weight3 = variable_with_weight_loss( [dim, 384], stddev=0.04, wl=0.004, name='weight3' )
    vi.variable_summaries( weight3 )
    bias3 = tf.Variable( tf.constant( 0.1, shape=[384] ), name='bais3' )
    vi.variable_summaries( bias3 )

    local3 = tf.nn.relu( tf.matmul( reshape, weight3 ) + bias3, name='local3_output' )

# [6] second fully connected layer
with tf.name_scope( '6_2nd_Fully_Connected_layer' ):
    # hidden layer: 192
    weight4 = variable_with_weight_loss( [384, 192], stddev=0.04, wl=0.004, name='weight4' )
    vi.variable_summaries( weight4 )
    bias4 = tf.Variable( tf.constant( 0.1, shape=[192] ) )
    vi.variable_summaries( bias4 )

    local4 = tf.nn.relu( tf.matmul( local3, weight4 ) + bias4, name='local4_output' )

# [7] third fully connected layer
with tf.name_scope( '7_3rd_Fully_Connected_layer' ):
    # output layer: 10
    weight5 = variable_with_weight_loss( [192, 10], stddev=1 / 192.0, wl=0.0, name='weight5' )
    vi.variable_summaries( weight5 )
    bias5 = tf.Variable( tf.constant( 0.0, shape=[10] ), name='bias5' )
    vi.variable_summaries( bias5 )

    final_output = tf.matmul( local4, weight5 ) + bias5


# define the loss function
def loss(image_input, label_input):
    images = image_input
    labels = tf.cast( label_input, tf.int64 )
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=images, labels=labels, name='cross_entropy_per_example' )
    cross_entropy_mean = tf.reduce_mean( cross_entropy, name='cross_entropy' )
    tf.add_to_collection( 'losses', cross_entropy_mean )
    return tf.add_n( tf.get_collection( 'losses' ), name='total_loss' )


with tf.name_scope( 'loss' ):
    # define the loss
    loss = loss( final_output, label_holder )
with tf.name_scope( 'train' ):
    # define the optimizer
    train_optimizer = tf.train.AdamOptimizer( 1e-3 ).minimize( loss )
    # calculate the accuracy
    label_output_match = tf.nn.in_top_k( final_output, label_holder, 1, name='batch_accuracy' )


# define the testing method
def model_test(images_test, labels_test, label_output_match, num):
    # evaluate the accuracy on testing batch
    num_examples = num

    # calculate the total number of batch in testing data set
    num_iter = int( math.ceil( num_examples / batch_size ) )
    total_sample_count = num_iter * batch_size
    # define the total matching number
    true_count = 0
    for iter_step in range( num_iter ):
        image_batch_test, label_batch_test = sess.run( [images_test, labels_test] )
        predictions = sess.run( [label_output_match],
                                feed_dict={image_holder: image_batch_test,
                                           label_holder: label_batch_test} )
        true_count += np.sum( predictions )

    precision = true_count / total_sample_count
    return precision


''' training session & testing session '''
# start training
with tf.name_scope( 'init' ):
    sess = tf.InteractiveSession()
    # merge all the former summary
    merged = tf.summary.merge_all()
    # store the corresponding graph data into tensor board
    writer = tf.summary.FileWriter( "Tensor_graph/", sess.graph )
    # initialize all the variables in model
    tf.global_variables_initializer().run()
    # start the image storing queue
    tf.train.start_queue_runners()
    # store the training model to desk (the most recent one)
    saver = tf.train.Saver( max_to_keep=1 )

# define the data storing array, for image plotting use
step_counter = []  # x axis store
precision_iter = []
loss_iter = []
examples_per_sec_iter = []
sec_per_batch_iter = []
accuracy_train = []
accuracy_test = []
# define the total size of training and testing batch
num_train = 50000
num_test = 10000
# set the x axis of training and testing related graph plotting
training_iter = 0
training_counter = []
batch_iter = int( math.ceil( num_train / batch_size ) )

# iteration (test accuracy of model during each iteration based on batch)
global_start_time = time.time()
for step in range( max_steps ):
    # define the start time of the training step
    start_time = time.time()

    # get the training image & corresponding label
    image_batch, label_batch = sess.run( [images_train, labels_train] )
    precision, accuracy_value, loss_value = sess.run( [label_output_match, train_optimizer, loss],
                                                      feed_dict={image_holder: image_batch,
                                                                 label_holder: label_batch} )
    # get the training precision each iteration
    precision_train = np.sum( precision ) / batch_size
    # calculate the duration time of each training iteration
    duration = time.time() - start_time

    # batch number processed per second
    examples_per_sec = batch_size / duration
    # total processing time for each batch
    sec_per_batch = float( duration )

    ''' store corresponding parameter into each storing array'''
    step_counter.append( step )
    precision_iter.append( precision_train )
    loss_iter.append( loss_value )
    examples_per_sec_iter.append( examples_per_sec )
    sec_per_batch_iter.append( sec_per_batch )

    ''' 
    # print training status of each batch training iteration 
    print( "step %d, accuracy=%.2f, loss=%.2f (%.1f examples/sec; %.3f sec/batch)"
           % (step, precision_train, loss_value, examples_per_sec, sec_per_batch) )
    '''

    ''' print training and test statues of each training iteration'''
    # plot graphs in the step of each training iteration
    # declearation of training_counter and batch_iter see in line 149
    if step % batch_iter == 0:
        print( "start one new training iteration plotting process ..." )
        training_iter += 1
        # calculate precision of training and testing session
        training_precision = model_test( images_train, labels_train, label_output_match, num_train )
        testing_precision = model_test( images_test, labels_test, label_output_match, num_test )
        ''' store corresponding parameter into each storing array'''
        training_counter.append( training_iter )
        accuracy_train.append( training_precision )
        accuracy_test.append( testing_precision )

        '''
        print( "****************************************************************************" )
        # print the training accuracy of training iteration
        print( 'training precision =%.3f' % training_precision )
        # print the testing accuracy of training iteration
        print( 'testing precision =%.3f' % testing_precision )
        print( "****************************************************************************" )
        '''

        # plot the training accuracy image of each batch iteration
        pl.plot_each( 'training accuracy per batch iteration', 'batch iteration',
                      'accuracy', step_counter, precision_iter,
                      'Batch_iteration/training accuracy .png' )
        # plot the training loss image of each batch iteration
        pl.plot_each( 'training loss per batch iteration', 'batch iteration',
                      'loss', step_counter, loss_iter,
                      'Batch_iteration/training loss .png' )
        # plot examples_per_sec image of each batch iteration
        pl.plot_each( 'training examples per second ', 'batch iteration',
                      'examples per second', step_counter, examples_per_sec_iter,
                      'Batch_iteration/training examples_per_sec .png' )
        # plot sec_per_batch image of each batch iteration
        pl.plot_each( 'training seconds per batch ', 'batch iteration',
                      'seconds per batch', step_counter, sec_per_batch_iter,
                      'Batch_iteration/training sec_per_batch .png' )
        print( "batch iteration based graphs finished" )

        # plot the training accuracy image of each training iteration
        pl.plot_each( 'training accuracy per training iteration', 'training iteration',
                      'accuracy', training_counter, accuracy_train,
                      'Training_iteration/training accuracy .png' )
        # plot the training accuracy image of each training iteration
        pl.plot_each( 'testing accuracy per training iteration', 'training iteration',
                      'accuracy', training_counter, accuracy_test,
                      'Training_iteration/testing accuracy .png' )
        print( "training iteration based graphs finished" )

    print( 'batch epoch', step, ': finished' )

# calculate the total execution time of the algorithm
global_end_time = time.time()
global_duration_time = global_end_time - global_start_time

# print the final training result
print( "************************************************************************" )
print( 'all %d batch epoches are done.' % (step + 1) )
print( 'final training accuracy: %.4f' % training_precision, 'final testing accuracy: %.4f' % testing_precision )
print( 'total training time is:%.2f' % global_duration_time )
# save the last epoch model
saver.save( sess, 'Tensor_model/raw_data_CNN', global_step=step + 1 )
