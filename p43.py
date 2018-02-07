import tensorflow as tf

X = tf.placeholder(tf.float32, [None, 3])


W = tf.Variable(tf.random_normal([3,2]))
b = tf.Variable(tf.random_normal([2,1]))

expr = tf.matmul(X,W)+b

sess = tf.Session()


x_data = [[1,2,3],[4,5,6]]
sess.run(tf.global_variables_initializer())

print("=== x_data===")
print(x_data)
print("=== W ===")
print(sess.run(W)) # 단순히 print(W)를 하면 W는 텐서이므로 (placeholder 이름, shape, 자료형) 형태로 출력될 것이므로 run한 tensor를 넣어야  내가 원하는 값을 출력할 수 있다.
print("=== b ===")
print(sess.run(b))
print("=== expr ===")
print(sess.run(expr, feed_dict = {X:x_data}))

sess.close()
