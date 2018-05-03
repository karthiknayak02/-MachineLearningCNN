import cv2
import os
import glob
from sklearn.utils import shuffle
import numpy as np

from tqdm import tqdm
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

import tensorflow as tf
import matplotlib.pyplot as plt

#Helper function to load in the training set of images and resize them all to the given size

def load_train(train_path, image_size, classes):
    images = []
    labels = []
    img_names = []
    cls = []

    print('Going to read training images')
    for fields in classes:   
        index = classes.index(fields)
        print('Now going to read {} files (Index: {})'.format(fields, index))
        path = os.path.join(train_path, fields, '*g')
        files = glob.glob(path)
        for fl in files:
            image = cv2.imread(fl)
            image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
            image = np.multiply(image, 1.0 / 255.0)
            images.append(image)
            label = np.zeros(len(classes))
            label[index] = 1.0
            labels.append(label)
            flbase = os.path.basename(fl)
            img_names.append(flbase)
            cls.append(fields)
    images = np.array(images)
    labels = np.array(labels)
    img_names = np.array(img_names)
    cls = np.array(cls)
    
    return images, labels, img_names, cls


#A class containing various information about the training set
class DataSet(object):

  def __init__(self, images, labels, img_names, cls):
    self._num_examples = images.shape[0]

    self._images = images
    self._labels = labels
    self._img_names = img_names
    self._cls = cls
    self._epochs_done = 0
    self._index_in_epoch = 0

  #Return the set of images
  def images(self):
    return self._images
	
  #Return the set of 1-hot class vectors
  def labels(self):
    return self._labels
	
  #Return the set of image filenames
  def img_names(self):
    return self._img_names
	
  #Return the set of class labels
  def cls(self):
    return self._cls
	
  #Return the number of examples in the training set
  def num_examples(self):
    return self._num_examples

  #Return the number of epochs that have been completed
  def epochs_done(self):
    return self._epochs_done

  #Retrieve the next batch of data to pass to the neural network
  #Inputs: 
  #batch_size: The number of training examples to return in a batch
  #Outputs:
  #the images in the next batch, the 1-hot class vectors for the next batch, the filenames in the next batch, and the class labels in the next batch
  def next_batch(self, batch_size):
    start = self._index_in_epoch
    self._index_in_epoch += batch_size

    if self._index_in_epoch > self._num_examples:
      self._epochs_done += 1
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch

    return self._images[start:end], self._labels[start:end], self._img_names[start:end], self._cls[start:end]


#Code to read in training data and put it in a decent format for learning
#Inputs: 
#train_path: a string containing the path to the training data
#image_size: image size (in pixels) that each training image will be resized to. Resulting dimensions will be image_size x image_size
#classes: an array containing each of the classes. For this assignment, it would be ['pembroke', 'cardigan']
#validation_size: Float corresponding to the proportion of the training set to set aside for validation. This is different than the test set!
#Returns:
#data_sets: a DataSet object containing images, labels, 1-hot label vectors, filenames, as well as training and validation data. 
def read_train_sets(train_path, image_size, classes, validation_size):
  class DataSets(object):
    pass
  data_sets = DataSets()

  images, labels, img_names, cls = load_train(train_path, image_size, classes)
  images, labels, img_names, cls = shuffle(images, labels, img_names, cls)

  if isinstance(validation_size, float):
    validation_size = int(validation_size * images.shape[0])


  validation_images = images[:validation_size]
  validation_labels = labels[:validation_size]
  validation_img_names = img_names[:validation_size]
  validation_cls = cls[:validation_size]

  train_images = images[validation_size:]
  train_labels = labels[validation_size:]
  train_img_names = img_names[validation_size:]
  train_cls = cls[validation_size:]

  data_sets.train = DataSet(train_images, train_labels, train_img_names, train_cls)
  data_sets.valid = DataSet(validation_images, validation_labels, validation_img_names, validation_cls)

  return data_sets

def load_test(test_path, image_size, classes):
    images = []
    labels = []
    img_names = []
    cls = []

    print('Going to read testing images')
    for fields in classes:
        index = classes.index(fields)
        print('Now going to read {} files (Index: {})'.format(fields, index))
        path = os.path.join(test_path, fields, '*g')
        files = glob.glob(path)
        for fl in files:
            image = cv2.imread(fl)
            image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
            image = np.multiply(image, 1.0 / 255.0)
            images.append(image)
            label = np.zeros(len(classes))
            label[index] = 1.0
            labels.append(label)
            flbase = os.path.basename(fl)
            img_names.append(flbase)
            cls.append(fields)
    images = np.array(images)
    labels = np.array(labels)
    img_names = np.array(img_names)
    cls = np.array(cls)

    return images, labels, img_names, cls

def read_test_sets(test_path, image_size, classes):

  images, labels, img_names, classes = load_test(test_path, image_size, classes)

  test_sets = DataSet(images, labels, img_names, classes)

  return test_sets



def make_model(data_sets, image_size):

    alpha = 0.001


    tf.reset_default_graph()

    convnet = input_data(shape=[None, image_size, image_size, 3], name='input')

    convnet = conv_2d(convnet, 32, 2, activation='relu')
    convnet = max_pool_2d(convnet, 2)

    convnet = conv_2d(convnet, 64, 2, activation='relu')
    convnet = max_pool_2d(convnet, 2)

    convnet = fully_connected(convnet, 1024, activation='relu')
    convnet = dropout(convnet, 0.8)

    convnet = fully_connected(convnet, 2, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=alpha, loss='categorical_crossentropy', name='targets')

    model = tflearn.DNN(convnet, tensorboard_dir='log')

    # if os.path.exists(model_name + '.meta'):
    #     model.load(model_name)
    #     print("Found existing moded, model loaded")

    return model


def main():

    train_path = "/Users/knayak/Spring/CS460G/New_HW6/data/training_data"
    test_path = "/Users/knayak/Spring/CS460G/New_HW6/data/testing_data"
    image_size = 50
    classes = ['pembroke', 'cardigan']
    validation_size = 0.20

    data_sets = read_train_sets(train_path, image_size, classes, validation_size)

    model = make_model(data_sets, image_size)

    # Training
    model_name = "corgi_10epochs"

    X = data_sets.train._images
    Y = data_sets.train._labels

    test_x = data_sets.valid._images
    test_y = data_sets.valid._labels

    model.fit({'input': X}, {'targets': Y}, n_epoch=21, validation_set=({'input': test_x}, {'targets': test_y}),
              snapshot_step=500, show_metric=True, run_id=model_name)

    # model.save(model_name)

    test_sets = read_test_sets(test_path, image_size, classes)

    total = len(test_sets._images)
    correct = 0


    for i in range(total):
        # pembroke: [1. 0.]
        # cardigan: [0. 1.]

        test_sets._images[i]

        # img = test_sets._images[i].reshape(image_size, image_size, 3)

        ans = model.predict([test_sets._images[i]])[0]
        # print(model.predict([img])[0])

        if np.argmax(ans) == 1: prediction = "cardigan"
        else: prediction = "pembroke"

        # print(prediction, test_sets._cls[i])
        if prediction == test_sets._cls[i]:
            correct += 1

    print("\nOverall accuracy over testing set:", correct*100.0/total)



    return
main()



