###############################################################################
# В процессе моих страданий я понял, что для генерации подходящего фона надо, #
# оказывается, знать сцену, где происходит событие. Данная простая нейросеть  #
# анализирует картинку, понимает её фон и выдаёт вердикт на основе классов из #
# датасета ADE20K. Написана при помощи моего собственного опыта.              #
###############################################################################

import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as l
import numpy as np

def create_model(inputs, shape, num_classes):
    model = tfk.models.Sequential([
        l.Input(shape=shape),
        l.Conv2D(32, (3, 3), activation='relu'),
        l.MaxPooling2D(2, 2),
        l.Conv2D(64, (3, 3), activation='relu'),
        l.MaxPooling2D((2, 2)),
        l.Flatten(),
        l.Dense(64, activation='relu'),
        l.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                 loss='categorial_crossentropy',
                 metrics='accurcy')
    
    return model
    
    