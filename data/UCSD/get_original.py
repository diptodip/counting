import sys
import os
from glob import glob
from tqdm import tqdm

image_files = [f for f in sorted(glob('A/*.png'))]
label_files = [f for f in sorted(glob('B/*.npy'))]

train_images = image_files[600:1400]
train_labels = label_files[600:1400]

test_images = image_files[0:600] + image_files[1400:]
test_labels = label_files[0:600] + label_files[1400:]

print('[i] getting training images')

for i in tqdm(range(len(train_images))):
    image = train_images[i]
    label = train_labels[i]
    os.system('cp {} A_original/'.format(image))
    os.system('cp {} B_original/'.format(label))

print('[i] getting testing images')

for i in tqdm(range(len(test_images))):
    image = test_images[i]
    label = test_labels[i]
    os.system('cp {} A_original_testing/'.format(image))
    os.system('cp {} B_original_testing/'.format(label))
