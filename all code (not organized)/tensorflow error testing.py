import tensorflow as tf
p = tf.data.Dataset.from_tensor_slices([(range(4),range(4),range(4))]).window(2)

for i in p:
  print(p)
