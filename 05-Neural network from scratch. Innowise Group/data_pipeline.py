import numpy as np
import idx2numpy
from sklearn.utils import shuffle


def load_data_mnist(path_images, path_labels, folder):
    
    images_array = idx2numpy.convert_from_file(folder + '/' + path_images)
    labels_array = idx2numpy.convert_from_file(folder + '/' + path_labels)

    return images_array, labels_array


def form_dataset(path_images, path_labels, folder_numbers, folder_fashion, ratio):
    
    images_numbers, labels_numbers = load_data_mnist(path_images, path_labels, folder_numbers)
    images_fashion, labels_fashion = load_data_mnist(path_images, path_labels, folder_fashion)
    
    images_fashion, labels_fashion = shuffle(images_fashion,
                                             labels_fashion, random_state=228)
    
    labels_fashion[labels_fashion > -1] = 10
    
    images = np.concatenate((images_numbers, images_fashion[:int(images_numbers.shape[0]*ratio)]), axis=0)
    labels = np.concatenate((labels_numbers, labels_fashion[:int(labels_numbers.shape[0]*ratio)]), axis=0)
    
    images, labels = shuffle(images, labels, random_state=228)
    
    return images, labels