import tensorflow
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import time
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from os.path import join
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator


num_classes = 2
resnet_weights_path = '/home/joseandresmr/Libraries/gml/Weights/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

my_new_model = Sequential()
my_new_model.add(ResNet50(include_top=False, pooling='avg', weights=resnet_weights_path))
my_new_model.add(Dense(num_classes, activation='softmax'))

# Indicate whether the first layer should be trained/changed or not.
my_new_model.layers[0].trainable = False

my_new_model.compile(optimizer='sgd', 
                     loss='categorical_crossentropy', 
                     metrics=['accuracy'])



image_size = 224
data_generator = ImageDataGenerator(preprocess_input)

train_generator = data_generator.flow_from_directory(
                                        directory='/home/joseandresmr/Libraries/gml/Datasets/hot-dog-not-hot-dog/train',
                                        target_size=(image_size, image_size),
                                        batch_size=10,
                                        class_mode='categorical')

validation_generator = data_generator.flow_from_directory(
                                        directory='/home/joseandresmr/Libraries/gml/Datasets/hot-dog-not-hot-dog/test',
                                        target_size=(image_size, image_size),
                                        class_mode='categorical')

# fit_stats below saves some statistics describing how model fitting went
# the key role of the following line is how it changes my_new_model by fitting to data
fit_stats = my_new_model.fit_generator(train_generator,
                                       steps_per_epoch=22,
                                       validation_data=validation_generator,
                                       validation_steps=1)



def read_and_prep_images(img_paths, img_height=image_size, img_width=image_size):
    imgs = [load_img(img_path, target_size=(img_height, img_width)) for img_path in img_paths]
    img_array = np.array([img_to_array(img) for img in imgs])
    output = preprocess_input(img_array)
    return(output)



hot_dog_image_dir = '/home/joseandresmr/Libraries/gml/Datasets/hot-dog-not-hot-dog/test/hot_dog'
hot_dog_paths = [join(hot_dog_image_dir, filename) for filename in
                            ['133012.jpg',
                             '133015.jpg']]

not_hot_dog_image_dir = '/home/joseandresmr/Libraries/gml/Datasets/hot-dog-not-hot-dog/test/not_hot_dog'
not_hot_dog_paths = [join(not_hot_dog_image_dir, filename) for filename in
                            ['6229.jpg',
                             '6261.jpg']]

img_paths = hot_dog_paths + not_hot_dog_paths
test_data = read_and_prep_images(img_paths)
preds = my_new_model.predict(test_data)

# for i, img_path in enumerate(img_paths):
#     plt.imshow(mpimg.imread(img_path))
#     print(preds[i])

import Image

for i, img_path in enumerate(img_paths):
    image = Image.open(img_path)
    image.show()
    print(preds[i])
    e = raw_input("next image")
    
