import numpy as np
import h5py
import os
import cv2
import random
import sys


def prepare_data(path):
    """
    To extract cuhk-03.mat file, extract all the images ndarray and save them in the directory
    :param path: the cuhk-03.mat file path
    :return: None
    """

    f = h5py.File('%s/cuhk-03.mat' % path)
    # ==============
    # f['labeled'] : <HDF5 dataset "labeled": shape (1, 5), type "|O">
    #
    # f['labeled'][0] : array([<HDF5 object reference>, <HDF5 object reference>,
    #                           <HDF5 object reference>, <HDF5 object reference>,
    #                           <HDF5 object reference>], dtype=object)
    #
    # [f['labeled'][0][i] for i in xrange(len(f['labeled'][0]))] :
    #                       [<HDF5 object reference>, <HDF5 object reference>,
    #                           <HDF5 object reference>, <HDF5 object reference>,
    #                           <HDF5 object reference>]
    #
    # m = [f['labeled'][0][i] for i in xrange(len(f['labeled'][0]))]
    # m[0] : <HDF5 object reference>
    # f[m[0]] : <HDF5 dataset "b": shape (10, 843), type "|O">
    # [f[m[0]][i] for i in xrange(len(f[m[0]]))] : shape(10, 843) 2-dims array
    # ==============

    labeled = [f['labeled'][0][i] for i in xrange(len(f['labeled'][0]))]
    labeled = [f[labeled[0]][i] for i in xrange(len(f[labeled[0]]))]
    detected = [f['detected'][0][i] for i in xrange(len(f['detected'][0]))]
    detected = [f[detected[0]][i] for i in xrange(len(f[detected[0]]))]
    datasets = [['labeled', labeled], ['detected', detected]]
    prev_id = 0

    for dataset in datasets:  # example : dataset : ['labeled', labeled]
        if not os.path.exists('%s/%s/train' % (path, dataset[0])):
            os.makedirs('%s/%s/train' % (path, dataset[0]))
        if not os.path.exists('%s/%s/val' % (path, dataset[0])):
            os.makedirs('%s/%s/val' % (path, dataset[0]))

        for i in xrange(0, len(dataset[1])):  # dataset[1] shape(10, 843)
            for j in xrange(len(dataset[1][0])):
                try:
                    # dataset[1] shape(10, 843); 0<i<10, 0<j<843
                    # dataset[1][i][j] shape(3-RGB-CHANNEL, WIDTH, HEIGHT) :
                    #       example{<HDF5 dataset "Pn": shape (3, 93, 295), type "|u1">}
                    image = np.array(f[dataset[1][i][j]]).transpose((2, 1, 0))
                    # image shape(HEIGHT, WIDTH, 3-RGB-CHANNEL)
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    image = cv2.imencode('.jpg', image)[1].tostring()
                    if len(dataset[1][0]) - j <= 100:
                        filepath = '%s/%s/val/%04d_%02d.jpg' % (path, dataset[0], j - prev_id - 1, i)
                    else:
                        filepath = '%s/%s/train/%04d_%02d.jpg' % (path, dataset[0], j, i)
                        prev_id = j
                    with open(filepath, 'wb') as image_file:
                        image_file.write(image)
                        print "Save (i:%d, j:%d) Image Success" % (i, j)
                except Exception as e:
                    continue


def get_pair(path, set, num_id, positive):
    """
    get a pair of positive or negative example, Must be a pair:
        1. randomly get/generate a random (id, id) pair between [0, num_id],
            if positive, get an pair with same [id], means two image from the same entity, only has diff posture
            if negative, get an pair with diff [ids], means two image from two entities, diff category
        2. randomly generate an entity instance between [0, 10] with the specified id: [id: 2]
        3. add the [id, id] pair image corresponding [file_path] into the result array: [pair]
        4. return the id pair
    :param path: root data file path
    :param set: "val" or "train" | dataset name
    :param num_id: whole entity number, using to generate randomly id of the pair
    :param positive: "True" or "False", if is positive image pair
    :return: a pair of image file path,
            [file_path_1, file_path_2]
    """
    pair = []
    if positive:
        value = int(random.random() * num_id)
        id = [value, value]
    else:
        while True:
            id = [int(random.random() * num_id), int(random.random() * num_id)]
            if id[0] != id[1]:
                break

    for i in xrange(2):
        filepath = ''
        while True:
            index = int(random.random() * 10)
            filepath = '%s/labeled/%s/%04d_%02d.jpg' % (path, set, id[i], index)
            if not os.path.exists(filepath):
                continue
            break
        pair.append(filepath)
    return pair


def get_num_id(path, set):
    """
    get all entities number of the Dataset (:param set)
    :param path: the dataset root path
    :param set: dataset name
    :return: the whole number or the length, entity number
    """
    files = os.listdir('%s/labeled/%s' % (path, set))
    files.sort()
    return int(files[-1].split('_')[0]) - int(files[0].split('_')[0]) + 1


def read_data(path, set, num_id, image_width, image_height, batch_size):
    """
    Get image positive and negative batch data with a mount of (:param batch_size)
    half of the (:param batch_size) is positive images and the remain half is negative images (with the same size)
    Also get image pair label:
        positive [1., 0.] ;
        negative [0., 1.] ;

    :param path: the data set root path
    :param set: the dataset category / dataset name | "val" or "train" | dataset name
    :param num_id: whole entity number, using to generate randomly id of the pair
    :param image_width: Target output width per image, using for imresize
    :param image_height: Target output height per image, using for imresize
    :param batch_size: batch size
    :return: return a list of +/- images and a list of corresponding labels

        1: training or validating image data list:
        [
            [positive_rgb_image_a_tensor_1, positive_rgb_image_a_tensor_2],
            [negative_rgb_image_b_tensor_1, negative_rgb_image_c_tensor_2],

            [positive_rgb_image_d_tensor_1, positive_rgb_image_d_tensor_2],
            [negative_rgb_image_e_tensor_1, negative_rgb_image_f_tensor_2],

            ...
        ]
        size: 2 * ([:param batch_size] // 2)

        after transpose(1,0,2,3,4) result will be:
        [
            [positive_rgb_img_a_tensor_1, negative_rgb_img_b_tensor_1, positive_rgb_img_d_tensor_1, negative_rgb_img_e_tensor_1, ...],
            [positive_rgb_img_a_tensor_2, negative_rgb_img_c_tensor_2, positive_rgb_img_d_tensor_2, negative_rgb_img_f_tensor_2, ...]
        ]
        shape: (2, batch_size, img_rows, img_cols, rgb_channels)

        2: training or validating image label list:
        [
            [1., 0.],
            [0., 1.],

            [1., 0.],
            [0., 1.],

            ...
        ]
        size: 2 * ([:param batch_size] // 2)

    """
    batch_images = []
    labels = []
    for i in xrange(batch_size // 2):  # floordiv of python

        # get POSITIVE target image filepath pair and NEGATIVE target image filepath pair
        pairs = [get_pair(path, set, num_id, True), get_pair(path, set, num_id, False)]
        for pair in pairs:
            images = []
            for p in pair:  # p is the pair of filepath
                # get image tensor, and add it to a array
                image = cv2.imread(p)
                image = cv2.resize(image, (image_width, image_height))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                images.append(image)
            # [images] : [rgb_image_tensor1, rgb_image_tensor2]
            # get image tensors array
            batch_images.append(images)
            # [batch_images] :
            #   [
            #       [positive_rgb_img_tensor_1, positive_rgb_img_tensor_2],
            #       [negative_rgb_img_tensor_1, negative_rgb_img_tensor_2],
            #   ]
        labels.append([1., 0.])
        labels.append([0., 1.])

    '''
    for pair in batch_images:
        for p in pair:
            cv2.imshow('img', p)
            key = cv2.waitKey(0)
            if key == 1048603:
                exit()
    '''
    return np.transpose(batch_images, (1, 0, 2, 3, 4)), np.array(labels)


if __name__ == '__main__':
    prepare_data(sys.argv[1])