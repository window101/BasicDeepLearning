
import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 

X = tf.placeholder(tf.float32, shape=[4,2])
Y = tf.placeholder(tf.float32, shape=[4,1])  
W1 = tf.Variable(tf.random_uniform([2,2]))
B1 = tf.Variable(tf.zeros([2]))
Z = tf.sigmoid(tf.matmul(X, W1) + B1)
W2 = tf.Variable(tf.random_uniform([2,1]))

B2 = tf.Variable(tf.zeros([1]))
Y_hat = tf.sigmoid(tf.matmul(Z, W2) + B2)

loss = tf.reduce_mean(-1*((Y*tf.log(Y_hat))+((1-Y)*tf.log(1.0-Y_hat))))
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

train_X = [[0,0],[0,1],[1,0],[1,1]]
train_Y = [[0],[1],[1],[0]]
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print("train data: "+ str(train_X))
    for i in range(20000):
        sess.run(train_step, feed_dict={X: train_X, Y: train_Y})
        if i % 5000 == 0:
            print('Epoch : ', i)
            print('Output : ', sess.run(Y_hat, feed_dict={X: train_X, Y: train_Y}))
        if i == 19999:
            print('W1 : ', sess.run(W1))
            print('B1 :', sess.run(B1))
            print('W2 :', sess.run(W2))
            print('B2 :', sess.run(B2))
    print('Final Output : ', sess.run(Y_hat, feed_dict={X: train_X, Y: train_Y})