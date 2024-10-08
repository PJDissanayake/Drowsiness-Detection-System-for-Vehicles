import tensorflow as tf

model = tf.keras.models.load_model('driving_behavior_model.h5')

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('driving_behavior_model.tflite', 'wb') as f:
    f.write(tflite_model)

print('Model successfully converted to Tensorflow lite')