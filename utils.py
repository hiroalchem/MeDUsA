import os
from pathlib import Path

import cv2
import numpy as np
import scipy.ndimage as ndi
from skimage import feature, img_as_float
from skimage.morphology import reconstruction
from skimage.segmentation import watershed
from skimage.draw import rectangle
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.losses import binary_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm


def dice_coeff(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score


def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss


def bce_dice_loss(y_true, y_pred):
    loss = binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss


def load_folder(data_d):
    p = Path(data_d)
    data_dirs = sorted(list(p.glob('./*_raw')))
    return data_dirs


def save_preprocessed_images(data_dir, temp_dir):
    images, filenames = load_x_gray_3slices(str(data_dir))
    os.makedirs(os.path.join(temp_dir, 'images', data_dir.name, 'images'))
    for idx, filename in enumerate(filenames):
        cv2.imwrite(os.path.join(temp_dir, 'images', data_dir.name, 'images',
                                 os.path.splitext((os.path.basename(filename)))[0] + '.png'), images[idx])


def load_x_gray_3slices(folder_path):
    print('load files')
    x_files = []
    for file in os.listdir(folder_path):
        base, ext = os.path.splitext(file)
        if ext == '.tif':
            x_files.append(file)
        else:
            pass

    x_files.sort()

    print(f'{len(x_files)} files')

    images = []
    for i, image_file in tqdm(enumerate(x_files)):
        if i == 0:
            image1 = cv2.equalizeHist(cv2.imread(folder_path + os.sep + x_files[i], cv2.IMREAD_GRAYSCALE))
            image0 = np.zeros_like(image1)
            image2 = cv2.equalizeHist(cv2.imread(folder_path + os.sep + x_files[i + 1], cv2.IMREAD_GRAYSCALE))
        elif i == (len(x_files) - 1):
            image0 = cv2.equalizeHist(cv2.imread(folder_path + os.sep + x_files[i - 1], cv2.IMREAD_GRAYSCALE))
            image1 = cv2.equalizeHist(cv2.imread(folder_path + os.sep + x_files[i], cv2.IMREAD_GRAYSCALE))
            image2 = np.zeros_like(image1)
        else:
            image0 = cv2.equalizeHist(cv2.imread(folder_path + os.sep + x_files[i - 1], cv2.IMREAD_GRAYSCALE))
            image1 = cv2.equalizeHist(cv2.imread(folder_path + os.sep + x_files[i], cv2.IMREAD_GRAYSCALE))
            image2 = cv2.equalizeHist(cv2.imread(folder_path + os.sep + x_files[i + 1], cv2.IMREAD_GRAYSCALE))
        image3ch = np.stack((image0, image1, image2), axis=2)
        images.append(image3ch)
        # image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    return images, x_files


def predict(data_dir, model, out_dir):
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_datagen.flow_from_directory(str(data_dir), target_size=(512, 512), class_mode=None,
                                                      shuffle=False, batch_size=1)
    filenames = test_generator.filenames
    nb_samples = len(filenames)
    y_pred = model.predict_generator(test_generator, nb_samples, verbose=1)
    os.makedirs(os.path.join(out_dir, data_dir.name), exist_ok=True)
    for i, y in enumerate(y_pred):
        # y = denormalize_y(y[:,:,-1]).astype('uint8')
        y = (y[:, :, 1] * 255.).astype('uint8')
        _, y = cv2.threshold(y, 127, 255, cv2.THRESH_BINARY)
        cv2.imwrite(os.path.join(out_dir, data_dir.name, os.path.basename(filenames[i])), y)


def load_images_and_masks(imps, pm, ft):
    """load images and predicted masks"""
    images = np.zeros((len(imps), 512, 512), np.uint8)
    masks = np.zeros((len(imps), 512, 512), np.uint8)
    for i in range(len(imps)):
        imr = cv2.imread(str(imps[i]), 0)
        if ft == "png":
            imm = cv2.imread(str((pm / imps[i].parent.name / imps[i].name).with_suffix(".png")), 0)
        else:
            imm = cv2.imread(str(pm / imps[i].parent.name / imps[i].name), 0)
        images[i] = imr
        masks[i] = imm
    return images, masks


def make_masked_images_from_array(images, masks):
    images_masked = np.zeros_like(images)
    for i in range(len(images)):
        images_masked[i] = cv2.bitwise_and(images[i], images[i], mask=masks[i])
    return images_masked


def blob_detection(images, masks, h, gauss, b, c, d):
    images3 = np.zeros_like(images)
    for i, image in enumerate(images):
        if gauss:
            image = ndi.gaussian_filter(image, sigma=gauss)
        else:
            pass
        image = img_as_float(image)
        h = h
        seed = image - h
        mask = image
        dilated = reconstruction(seed, mask, method='dilation')
        image = image - dilated
        image = (image * 255).astype('uint8')
        images3[i] = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, b, c)
    images4 = ndi.gaussian_filter(images3, sigma=1)
    images4 = (images4 >= 127).astype('uint8')
    images4 = make_masked_images_from_array(images4, masks)
    distance = ndi.distance_transform_edt(images4)
    local_max = feature.peak_local_max(distance, indices=False, labels=images4, min_distance=d,
                                       exclude_border=False)
    markers = ndi.label(local_max, structure=np.ones((3, 3, 3)))[0]
    labels = watershed(-distance, markers, mask=images4)
    return labels


def count_blob4(images, masks, h=0.6, gauss=None, b=31, c=-4, d=4, v=24):
    labels = blob_detection(images, masks, h, gauss, b, c, d)
    if labels.max() > 1500:
        return 0
    else:
        _, nums = np.unique(labels, return_counts=True)
        return np.count_nonzero(nums > v) - 1


def draw_blob4(imps, images, masks, out_dir, h=0.6, gauss=None, b=31, c=-4, d=4, v=24):
    labels = blob_detection(images, masks, h, gauss, b, c, d)
    if labels.max() > 1500:
        return 0
    else:
        uni, nums = np.unique(labels, return_counts=True)
        centers = []
        for i in range(len(uni) - 1):
            if nums[i + 1] > v:
                centers.append(ndi.measurements.center_of_mass(labels == uni[i + 1]))
            else:
                pass
        dr = np.zeros_like(images)
        for c in centers:
            z, raw, col = c
            start = (int(z) - 2, int(raw) - 3, int(col) - 3)
            end = (int(z) + 2, int(raw) + 3, int(col) + 3)
            z, r, c = rectangle(start, end, shape=dr.shape)
            dr[z, r, c] = 255

        for i in range(len(dr)):
            os.makedirs(os.path.join(out_dir, imps[i].parent.name), exist_ok=True)
            cv2.imwrite(os.path.join(out_dir, imps[i].parent.name, imps[i].name), dr[i])

        return len(centers)
