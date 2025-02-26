import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
import tensorflow as tf
import os

img_height = 64
img_width = 64
batch_size = 32

with open("history_xray", "rb") as pf:
    ann_model = pickle.load(pf)

# Accuracy ​
plt.figure()
plt.subplot(1,2,1)
plt.plot(ann_model.history['accuracy'])
plt.plot(ann_model.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training set', 'Validation set'], loc='upper left')
plt.savefig('train_accuracy.png')
plt.subplot(1,2,2)
plt.plot(ann_model.history['val_loss'])
plt.plot(ann_model.history['loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training set', 'Test set'], loc='upper left')
plt.savefig('train_loss.png')
plt.show(block=False)

model = tf.keras.models.load_model('penumonia.keras')
model.summary()

path = '../99_vision_pneumonia/'
train_path = 'chest_xray/train/'
test_path = 'chest_xray/test/'
val_path = 'chest_xray/val/'

test_n = os.path.join(path, test_path, 'NORMAL/')
test_p = os.path.join(path, test_path, 'PNEUMONIA/')

img_list_n = os.listdir(test_n)
img_list_p = os.listdir(test_p)

test_datagen = ImageDataGenerator(rescale = 1./255,
                                shear_range = 0.2,
                                zoom_range = 0.2,
                                horizontal_flip = True)

test_ds = test_datagen.flow_from_directory(os.path.join(path, train_path),
                                                target_size = (img_height, img_width),
                                                batch_size = batch_size,
                                                class_mode = 'binary')


