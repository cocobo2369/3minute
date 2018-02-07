import tensorflow as tf
import numpy as np

data = np.loadtxt('./data.csv', dtype = 'float32', delimiter = ',',unpack = True)
#unpack으로 행과 열을 바꾸는 이유는 ? 메모리 상 행들이 순서대로 묶여있고 그 안에 열들이 있다.
#내가 원하는 것은 열들인데 열들은 메모리가 띄엄띄엄 떨어져 있으니 행을 열로, 열을 행으로 바꾸면
#띄엄띄엄 떨어져 있던 열들이 한데묶인 행이 되어 데이터를 구분하기 쉬워진다.
x_data = np.transpose(data[0:2])
y_data = np.transpose(data[2:])
#이렇게 data[0:2] 로 하면 실제 값은 0,1 행을 가져가는 거지만 unpack=True를 하여 행으로 바뀐 열을 내가 갖는 것이다
#그리고 다시 transpose 하여 행이 열이 된다.
#정리하면 열 -> 행 -> 열 화 한 것

#학습횟수를 카운트 하는 변수 , 학습에 직접사용되는 것이 아니라(trainable = False) 학습횟수를 계속 누적시킴
global_step = tf.Variable(0, trainable = False, name = 'global_step')

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_uniform([2,10],-1,1))
L1 = tf.matmul(X,W1)
L1 = tf.nn.relu(L1)

W2 = tf.Variable(tf.random_uniform([10,20],-1,1))
L2 = tf.matmul(L1,W2)
L2 = tf.nn.relu(L2)

W3 = tf.Variable(tf.random_uniform([20,3],-1,1))
model = tf.matmul(L2, W3)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = Y, logits = model))

optimizer = tf.train.AdamOptimizer(learning_rate = 0.1)
train_op = optimizer.minimize(cost, global_step = global_step) #여기서 학습횟수를 기억한다 그래서 1씩 늘린다

#돌리기 with 이전 자료가 있으면 불러들이기
sess = tf.Session()
saver = tf.train.Saver(tf.global_variables())

#checkPoint
ckpt = tf.train.get_checkpoint_state('./model')
if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    saver.restore(sess,ckpt.model_checkpoint_path)
else :
    sess.run(tf.global_variables_initializer())

for step in range(2) :
    sess.run(train_op, feed_dict ={X:x_data, Y:y_data})

    print("step : %d," %sess.run(global_step),
            "cost : %.3f" %sess.run(cost, feed_dict={X:x_data, Y:y_data}))

saver.save(sess,'./model/dnn.ckpt', global_step = global_step)


prediction = tf.argmax(model, 1)
target = tf.argmax(Y,1)

print("예측값 :", sess.run(prediction, feed_dict={X:x_data}))
print("실제값 :", sess.run(target, feed_dict ={X:x_data, Y:y_data}))

is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print("정확도 : %.2f" %sess.run(accuracy*100,feed_dict={X:x_data, Y:y_data}))
