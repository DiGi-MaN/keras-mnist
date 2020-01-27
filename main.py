from mnist_cnn import TextRecognizer
import os
import fnmatch
import random
import matplotlib.pyplot as plt
from keras.preprocessing import image
import numpy as np

cnn = TextRecognizer(False)

matches = []
for root, dirnames, filenames in os.walk('tests'):
    for filename in fnmatch.filter(filenames, '*.png'):
        matches.append(os.path.join(root, filename))

random.shuffle(matches)
results = matches[:25]

plt.figure(figsize=(5, 5))

for i in range(25):
    res = cnn.predict(results[i])

    test_img = image.load_img(results[i], target_size=(28, 28), color_mode='grayscale')
    test_img = image.img_to_array(test_img)
    test_img = np.expand_dims(test_img, axis=0)
    test_img = test_img.reshape(28, 28)

    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_img)
    plt.xlabel(str(res))
plt.show()