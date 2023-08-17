0###################################################
# Тут собраны разные утилиты для масштабирования, #
# нормализации и прочих взаимодействий.           #
###################################################

import cv2
import tqdm
import pandas as pd
import base64
import numpy as np
import operator

def load_images(images_p, seg_images_p, target_size):
    """
    Загружает картинки из путей и меняет размер
    :param list images_p список путей картинок
    :param list seg_images_p список путей сегментированных картинок
    :param tuple target_size кортеж с желаемым размером
    :return list images список самих картинок
    :return list seg_images список сегм. картинок
    """
    images = []
    seg_images = []
    for _, (image, seg_image) in tqdm.tqdm(enumerate(zip(images_p, seg_images_p)), total=len(images_p), desc='Загружаю из пути...'):
        img = cv2.imread(image)[:,:,::-1]
        seg = cv2.imread(seg_image)[:,:,::-1]
        img = cv2.resize(img, target_size)
        seg = cv2.resize(seg, target_size)
        images.append(img)
        seg_images.append(seg)
    return images, seg_images
    
# def img_resize(images, seg_images, target_size):
#     """
#     Функция для изменения размеров.
#     :param list images список картинок
#     :param list seg_images список сегм. картинок
#     :return list images_rsz список самих картинок, но с новым размером
#     :return list seg_images_rsz список сегм. картинок, но с новым размером
#     """
#     images_rsz = []
#     seg_images_rsz = []
#     for _, (image, seg_image) in tqdm.tqdm(enumerate(zip(images, seg_images)), total=len(images), desc='Меняю размер...'):
#         img = cv2.resize(image, target_size)
#         seg_img = cv2.resize(seg_image, target_size)
#         images_rsz.append(img)
#         seg_images_rsz.append(seg_img)
#     return images_rsz, seg_images_rsz

# Блок для сохранения картинок в csv и загрузки их оттуда.

def image_to_base64(images):
    """
    Функция-кодировщик. Позволяет перевести изображения в более лёгкий формат для дальнейшего сохранения
    :param list image_p список картинок
    :param list seg_image список сегм. картинок
    """
    encoded_strings = []
    encoded_seg_strings = []
    for _, (image, seg_image) in tqdm.tqdm(enumerate(zip(images[0], images[1])), total=len(images[0]), desc='Кодирую...'):
        encoded_string = base64.b64encode(cv2.imencode('.jpg', image)[1]).decode('utf-8')
        encoded_seg_string = base64.b64encode(cv2.imencode('.png', seg_image)[1]).decode('utf-8')
        encoded_strings.append(encoded_string)
        encoded_seg_strings.append(encoded_seg_string)
    return encoded_strings, encoded_seg_strings
    
def base64_to_img(enc_strings, enc_seg_strings):
    images = []
    seg_images = []
    for _, (enc_str, enc_seg_str) in tqdm.tqdm(enumerate(zip(enc_strings, enc_seg_strings)), total=len(enc_strings), desc='Декодирую...'):
        decoded_bytes = base64.b64decode(enc_str)
        decoded_seg_bytes = base64.b64decode(enc_seg_str)
        decoded_image = cv2.imdecode(np.frombuffer(decoded_bytes, np.uint8), cv2.IMREAD_COLOR)
        decoded_seg_image = cv2.imdecode(np.frombuffer(decoded_seg_bytes, np.uint8), cv2.IMREAD_COLOR)
        images.append(decoded_image)
        seg_images.append(decoded_seg_image)
    return images, seg_images
        

def img_normilize(images):
    norm_images = []
    for img in tqdm.tqdm(images, total=len(images), desc='Нормализация...'):
        img_norm = img/255.0
        norm_images.append(img_norm)
    return norm_images


def one_hot_it(seg_images):
    one_hot_seg_images = []
    for img in tqdm.tqdm(seg_images, total=len(seg_images), desc='Изменение в вектор...'):
        w = img.shape[0]
        h = img.shape[1]
        x = np.zeros((w, h, 3688))
        for i in range(0, w):
            for j in range(0, h):
                x[i, j, img[i, j]] = 1
        one_hot_seg_images.append(x)
    return one_hot_seg_images
    
        
    