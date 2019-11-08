import tensorflow as tf

tf.reset_default_graph()

with tf.name_scope('Operaciones'):
    with tf.name_scope('Escopo_A'):
        a = tf.add(2,2,name = 'add')
        with tf.name_scope('Escopo_B'):
            b = tf.multiply(a, 3, name = 'multi1')
            c = tf.multiply(b, a, name = 'multi1')

with tf.Session() as sess:
    writer = tf.summary.FileWriter('output',sess.graph)
    print(sess.run(c))
    writer.close

tf.get_default_graph()

grafo1 = tf.get_default_graph()

grafo1

grafo2 = tf.Graph()

grafo2

with grafo2.as_default():
    print(grafo2 is tf.get_default_graph())