TensorFlow Code

import tensorflow as tf
import numpy as np


if tf.config.list_physical_devices('GPU'):
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


X = tf.random.normal((1000, 10))  
y = tf.random.uniform((100,), minval=0, maxval=3, dtype=tf.int32)  


class MLP(tf.keras.Model):
    def __init__(self):
        super(MLP, self).__init__()
        self.layer1 = tf.keras.layers.Dense(32, input_shape=(10,))
        self.layer2 = tf.keras.layers.Dense(3, input_shape=(32,))  
        
    def call(self, x):
        x = self.layer1(x)
        x = tf.nn.relu(x)
        x = self.layer2(x)  
        return x


model = MLP()
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)


for epoch in range(10):
    with tf.GradientTape() as tape:
        outputs = model(X)
        loss = loss_fn(y, outputs)  
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    if epoch % 2 == 0:
        print(f"Epoch {epoch}, Loss: {loss.numpy()}")

# Test model
test_input = tf.random.normal((1, 10))
test_output = model(test_input)
print("Predicted class:", tf.argmax(test_output, axis=1).numpy())
