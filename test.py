import tensorflow as tf
a = tf.constant([1], shape=[10,10])
with tf.Session() as sess:
    print(sess.run(a))