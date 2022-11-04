import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
if __name__=="__main__":
    print("Hello World")
    print(hub.__version__)
    m = tf.keras.Sequential([
    hub.KerasLayer("https://tfhub.dev/google/imagenet/inception_resnet_v2/feature_vector/5",
                   trainable=False),  # Can be True, see below.
    tf.keras.layers.Dense(4, activation='softmax')])
    input_tensor = tf.convert_to_tensor(np.random.rand(1,299,299,3))
    print()