import os
import sys
import warnings

import numpy as np

import matplotlib.pyplot as plt

from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from scipy.misc import imresize
from skimage.morphology import label
from PIL import Image



# Set some parameters
IMG_WIDTH = 244
IMG_HEIGHT = 224
IMG_CHANNELS = 3

# We could generalize later on the whole competition dataset.
TRAIN_PATH = './train/'
LABEL_PATH = './train_label/'
TEST_PATH = './test/'

#print(os.listdir("../input/"))
warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
#seed = 77
#random.seed = seed
#np.random.seed = seed


# Get train and test IDs
train_ids = next(os.walk(TRAIN_PATH))[2]
label_ids = next(os.walk(LABEL_PATH))[2]
test_ids = next(os.walk(TEST_PATH))[2]

# Get and resize train images and masks
X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)

# dtype=np.float32
features_im_path = []
#features_im_path.append(path)

print('Getting and resizing train images and masks ... ')
sys.stdout.flush()
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    print(id_)
    path = TRAIN_PATH + id_
    img = imread(path)[:,:,:IMG_CHANNELS]
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_train[n] = img

np.save("x_train.npy", X_train)

for n, id_ in tqdm(enumerate(label_ids), total=len(label_ids)):
    print(id_)
    path = LABEL_PATH + id_
    img = imread(path, as_grey=True)
    img = np.reshape(np.array(img), (img.shape[0], img.shape[1])) # To transform as an numpy array first
    #img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    img = imresize(img, (IMG_HEIGHT, IMG_WIDTH), mode='L', interp='nearest')
    img = np.reshape(img,(img.shape[0], img.shape[1], 1))
    Y_train[n] = img

np.save("y_train.npy", Y_train)

# Get and resize test images
X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)

sizes_test = []

print('Getting and resizing test images ... ')
sys.stdout.flush()
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):

    path = TEST_PATH + id_
    img = imread(path)[:,:,:IMG_CHANNELS]
    sizes_test.append([img.shape[0], img.shape[1]])
    print(id_,[img.shape[0], img.shape[1]])
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_test[n] = img

np.save("x_test.npy", X_test)
np.save("sizes_test.npy", sizes_test)

print("X_train: ",X_train.shape)
print("Y_train: ",Y_train.shape)
print("X_test: ",X_test.shape)
print('Done!')

