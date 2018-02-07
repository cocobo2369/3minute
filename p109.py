import tensorflow as tf

#텐서플로 내장 모듈 import
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot = True)

# 그림 데이터 28*28 = 784픽셀 = 특징
# 숫자 0~9 10개 = 분류

X = tf.placeholder(tf.float32, [None, 784]) # float arr[][] 같은 느낌 순서로
Y = tf.placeholder(tf.float32, [None, 10])
#데이터가 너무 크면 메모리 부하가 높아지므로 적당한 크기로 잘라서 학습시킴 = 미니배치
#None 으로 주면 텐서플로우가 알아서 미니배치화함


keep_prob = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_normal([784,256], stddev = 0.01)) #표준편차가 0.01 인 정규분포
L1 = tf.nn.relu(tf.matmul(X,W1))
L1 = tf.nn.dropout(L1,keep_prob)

W2 = tf.Variable(tf.random_normal([256,256], stddev = 0.01))
L2 = tf.nn.relu(tf.matmul(L1,W2))
L2 = tf.nn.dropout(L2, keep_prob)

W3 = tf.Variable(tf.random_normal([256,10], stddev = 0.01))
model = tf.nn.relu(tf.matmul(L2, W3))


cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = Y, logits = model))

optimizer = tf.train.AdamOptimizer(learning_rate = 0.001)
train_op = optimizer.minimize(cost)

sess = tf.Session()
init = tf.global_variables_initializer()

sess.run(init)

batch_size = 100
total_batch = int(mnist.train.num_examples / batch_size)

for epoch in range(30) :
    total_cost = 0

    for i in range(total_batch) :
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)

        _, cost_val = sess.run([train_op, cost], feed_dict={X:batch_xs, Y:batch_ys, keep_prob : 0.8})
        total_cost += cost_val

    print('Epoch:', "%04d" %(epoch +1), "Avg Cost = ","{:.3f}" .format(total_cost/total_batch))

print("완료")

is_correct = tf.equal(tf.argmax(model,1),tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print("정확도 :", sess.run(accuracy, feed_dict={X:mnist.test.images, Y: mnist.test.labels, keep_prob : 1}))
