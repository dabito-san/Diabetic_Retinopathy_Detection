from PIL import Image
import numpy as np
import cv2
import os
from multiprocessing import Pool
from tqdm import tqdm


def trim_background(image):
    """
    Converts image to grayscale, then computes a binary matrix of the
    pixels that are above a certain threshold of the mean pixel values
    of the image. Then, obtains the first (and last) row (and column)
    where a certain percentage of the pixels are not black (zero value).
    Min/Max rows/cols obtained are used to crop the image.
    :param image: PIL image
    :return: a PIL image
    """
    percentage = 0.02

    image = np.array(image)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # pixels where value is higher than 10% of the mean of non-black pixels
    img = image_gray > 0.1 * np.mean(image_gray[image_gray != 0])
    # count of zero-value pixels
    row_sums = np.sum(img, axis=1)
    col_sums = np.sum(img, axis=0)
    # row/col indexes where most of the pixel values are non-zero.
    rows = np.where(row_sums > img.shape[1] * percentage)
    cols = np.where(col_sums > img.shape[0] * percentage)
    # row/col indexes to keep after image crop
    min_row, min_col = np.min(rows), np.min(cols)
    max_row, max_col = np.max(rows), np.max(cols)

    image_crop = image[min_row: max_row + 1, min_col: max_col + 1]

    return Image.fromarray(image_crop)


def resize_image(image, output_size):
    """
    Resizes an image to a size [output_size, output_size].
    :param image: PIL image
    :param output_size: Int desired new size
    :return: PIL image
    """
    old_size = image.size
    ratio = float(output_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    img = image.resize(new_size, Image.ANTIALIAS)
    new_img = Image.new('RGB', (output_size, output_size))
    new_img.paste(img, ((output_size - new_size[0]) // 2, (output_size - new_size[1]) // 2))

    return new_img


def save_resized_image(args):
    image_file, image_path, output_path, output_size = args
    image = Image.open(os.path.join(image_path, image_file))
    image = trim_background(image)
    image = resize_image(image, output_size)
    image.save(os.path.join(output_path, image_file))


def save_resized_image_multiprocess(image_path, output_path, output_size):
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    jobs = [
        (file, image_path, output_path, output_size)
        for file in os.listdir(image_path)
        if os.path.isfile(os.path.join(image_path, file))
    ]

    with Pool() as p:
        list(tqdm(p.map(save_resized_image, jobs)))


if __name__ == '__main__':
    image_path = '../data/train/'
    output_path = '../data/train/resized/'
    output_size = 300
    for file in os.listdir(image_path):
        if not (file.startswith('.')):
            if os.path.isfile(os.path.join(image_path, file)):
                save_resized_image((file, image_path, output_path, output_size))

    # TODO resize test images
    # save_resized_image_multiprocess('../data/test/',
    #                                 '../data/test/resized/')
