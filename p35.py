import tensorflow as tf

hello = tf.constant("hello! This is first time of tensorflow!") ## string 형 상수
print(hello)


a = tf.constant(3) #int 형 상수
b = tf.constant(5)
c = tf.add(a,b)
print(c)

sess = tf.Session()
print(sess.run(hello))
print(sess.run([hello, a,b,c]))
print(sess.run(c))

sess.close()
