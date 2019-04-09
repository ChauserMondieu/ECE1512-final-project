import tensorflow as tf


# calculate the standard deviation & variance of each scalar (variable in CNN)
def variable_summaries(var):
    # count the average of the variable
    with tf.name_scope("summaries"):
        mean = tf.reduce_mean(var)
        tf.summary.scalar("mean",mean)
    # calculate the standard deviation of variable
    with tf.name_scope("stddev"):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar("stddev",stddev)
    # count the min and max value of variable
    tf.summary.scalar("max",tf.reduce_max(var))
    tf.summary.scalar("min",tf.reduce_min(var))
    # show the distribution of variable with histogram
    tf.summary.histogram("histogram",var)
