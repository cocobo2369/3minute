import tensorflow as tf
import numpy as np

#특징
x_data = np.array([[0,0],
                   [1,0],
                   [1,1],
                   [0,0],
                   [0,0],
                   [0,1]])
#분류
y_data = np.array([[1,0,0],
                   [0,1,0],
                   [0,0,1],
                   [1,0,0],
                   [1,0,0],
                   [0,0,1]])
#그래프 작성
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_uniform([2,10], -1,1))
b1 = tf.Variable(tf.zeros([10]))

W2 = tf.Variable(tf.random_uniform([10,3],-1,1))
b2 = tf.Variable(tf.zeros([3]))

#1 선형회기 함수
L1 = tf.add(tf.matmul(X,W1),b1)
#1-2 활성화 함수
L1 = tf.nn.relu(L1)

#2 선형회기 함수
model = tf.add(tf.matmul(L1,W2),b2) #출력층은 활성화 함수를 보통 사용하지 않는다

#3 손실 함수 - 교차 엔트로피(label이 기준이고 logits 가 비교대상인듯)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = Y, logits = model))

#4 학습
optimizer = tf.train.AdamOptimizer(learning_rate = 0.1)
train_op = optimizer.minimize(cost) #돌려야할 대상



# 돌리기
sess = tf.Session()
#1  Variable 초기화
init = tf.global_variables_initializer()
sess.run(init)

for step in range(100):
    sess.run(train_op,feed_dict = {X:x_data, Y:y_data})

    if (step+1)%10 == 0 :
        print(step+1, sess.run(cost,feed_dict={X:x_data , Y:y_data}))

#결과 확인하기
prediction = tf.argmax(model, 1) #함수가 예측한 값
target = tf.argmax(Y,1)
print("예측값:", sess.run(prediction,feed_dict={X:x_data}))
print("실제값:", sess.run(target,feed_dict={Y:y_data}))

is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print("정확도: %.2f" %sess.run(accuracy*100, feed_dict = {X:x_data,Y:y_data}))


