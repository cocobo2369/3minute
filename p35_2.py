import tensorflow as tf

#placeholder 입력으로 값 출력하기

a = tf.placeholder(tf.float32, [])
b = tf.placeholder(tf.float32, [])

c = tf.add(a,b)

sess = tf.Session()
print(sess.run(c,feed_dict={a:1,b:4}))

sess.close()

