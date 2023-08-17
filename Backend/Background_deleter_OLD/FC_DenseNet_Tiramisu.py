######################################################################################################
# Отныне и впредь я воссоздаю эту архитектуру модели.                                                #
# Выбор пал на неё из-за её превосходства над другими моделями сегментации.                          #
# Данная модель способна отлично сегментировать разные данные.                                       #
# Модель будет перенесена и адаптирована мной для улучшения выполнения задачи. Спасибо за внимание!  #
# Код для переработки был взят из этого источника (https://github.com/ShashiAI/FC-DenseNet-Tiramisu) #
######################################################################################################

# from __future__ import division
# import os,time
# import tensorflow as tf
# import tensorflow.keras as tfk
# import tensorflow.keras.layers as l
# import numpy as np

# def preact_conv(inputs, n_filters, filter_size=(3, 3), dropout_p=0.2):
#     """
#     Стандартный предактивационный слой для DenseNets-а, 
#     Добавляет пакетную нормализацию, нелинейную ReLU, свёртку и выпадение. 
#     """

#     norm = l.BatchNormalization()(inputs)
#     preact = l.Activation('relu')(norm)
#     conv = l.Conv2D(n_filters, filter_size, padding='same')(preact)
#     if dropout_p != 0.0:
#         conv = l.Dropout(1.0-dropout_p)(conv)
#     return conv


# def DenseBlock(stack, n_layers, growth_rate, dropout_p, scope=None):
#     """
#     Полносвязный блок для нейросети.
#     :param stack входной четырёхмерный тензор
#     :param n_layers число внутренних слоёв
#     :param growth_rate число карт признаков на внутренний слой
#     :return stack четырёхмерный тензор новых карт признаков
#     :return new_features четырёхмерный тензор, содержащий ТОЛЬКО новые карты признаков из этого блока
#     """
    
#     with tf.name_scope(scope) as sc:
#         new_features = []
#         for i in range(n_layers):
#             # Подсчёт новых карт признаков
#             layer = preact_conv(stack, growth_rate, dropout_p=dropout_p)
#             new_features.append(layer)
#             # Укладывание нового слоя
#             stack = tf.concat([stack, layer], axis=-1)
#         new_features = tf.concat(new_features, axis=-1)
# #         return stack, new_features
#         return stack
    

# # def TransitionLayer(inputs, n_filters, dropout_p=0.2, compression=1.0, scope=None):
# #     """
# #     Переходный слой
# #     Добавляет преактивационный блок и 2х2 пуллинг
# #     """
    
# #     with tf.name_scope(scope) as sc:
# #         if compression < 1.0:
# #             n_filters = tf.to_int32(tf.floor(n_filters*compression))
# #         l = preact_conv(inputs, n_filters, filter_size=(1, 1), dropout_p=dropour_p)
# #         l = l.AveragePooling2D((2, 2), strides=(2, 2))(l)
# #         return l
    

# def TransitionDown(inputs, n_filters, scope=None):
#     """
#     Слой перехода вниз
#     Преактивационный слой + 2х2 выбор максимума
#     """
    
# #     with tf.name_scope(scope) as sc:
# #         print('Принимаемый размер' + inputs.output_shape)
#     t_d = l.BatchNormalization()(inputs)
#     t_d = l.Activation('relu')(t_d)
#     t_d = l.Conv2D(n_filters, 1, padding='same')(t_d)
#     t_d = l.MaxPooling2D(2)(t_d)
#     return t_d

    

# def TransitionUp(block_to_upsample, skip_connection, n_filters_keep, scope=None):
#     """
#     Слой перехода вверх
#     Производит повышающую дискретизацию на block_to_unsample с фактором 2 и складывает с skip_connetion
#     """
    
#     with tf.name_scope(scope) as sc:
#         # ПОвышаюющая дискретизация

#         t_u = l.Conv2DTranspose(n_filters_keep, kernel_size=(3, 3), strides=(2, 2), padding='same')(block_to_upsample)
#         # Сложение с skip_connection
# #         t_u = tf.concat([t_u, skip_connection], axis=-1)
#         return t_u
    

# def build_fc_densenet(inputs, preset_model='FC-DenseNet56', num_classes=3688, n_filters_first_conv=48, n_pool=5, growth_rate=12, n_layers_per_block=4, dropout_p=.2, scope=None):
#     """
#     Создание итоговой модели
#     :param str preset_model пресетная модель (FC-DenseNet56, FC-DenseNet67, FC-DenseNet103)
#     :param int n_classes число классов
#     :param int n_filters_first_conv число фильтров для первой свёртки
#     :param int n_pool число слоёв пуллинга = слоёв перехода вниз/вверх
#     :param int growth_rate число создаваемых новых карт признаков в слое за блок
#     :param n_layers_per_block число слоёв на блок. Может быть числом или списком размером 2 * n_pool+1
#     :param float dropout_p коэффициент выпадения для каждой свёртки (0 если не нужен)
#     """
    
#     if preset_model == 'FC-DenseNet56':
#         n_pool=5
#         growth_rate=12
#         n_layers_per_block=4
#     elif preset_model == 'FC-DenseNet67':
#         n_pool=5
#         growth_rate=16
#         n_layers_per_block=5
#     elif preset_model == 'FC-DenseNet103':
#         n_pool=5
#         growth_rate=16
#         n_layers_per_block=[4, 5, 7, 10, 12, 15, 12, 10, 7, 5, 4]
        
#     if type(n_layers_per_block) == list:
#         assert (len(n_layers_per_block) == 2 * n_pool + 1)
#     elif type(n_layers_per_block) == int:
#         n_layers_per_block = [n_layers_per_block] * (2 * n_pool + 1)
#     else:
#         raise ValueError
        
#     with tf.compat.v1.variable_scope(scope, preset_model, [inputs]) as sc:
        
#         # Первая свёртка #
#         stack = l.Conv2D(n_filters_first_conv, (3, 3), name='first_conv', padding='same')(inputs)
#         n_filters = n_filters_first_conv
        
#         # Понижаем дискретизацию #
#         skip_connection_list = []
        
#         for i in range(n_pool):
#             # Полносвязный блок
            
# #             stack, _ = DenseBlock(stack, n_layers_per_block[i], growth_rate, dropout_p, scope='denseblock%d'%(i+1))
#             stack = DenseBlock(stack, n_layers_per_block[i], growth_rate, dropout_p, scope='denseblock%d'%(i+1))
            
#             n_filters += growth_rate * n_layers_per_block[i]
#             # Под конец полносвязного блока текущий стек сохраняется в skip_connetions_list
# #             skip_connection_list.append(stack)
#             # Переход ниже
            
#             stack = TransitionDown(stack, n_filters, scope='transitiondown%d'%(i+1))
        
# #         skip_connection_list = skip_connection_list[::-1]
        
#         # Дно нейросети #
        
#         # Мы будем поднимать только новые карты признаков
# #         stack, block_to_upsample = DenseBlock(stack, n_layers_per_block[n_pool], growth_rate, dropout_p, scope='denseblockbn%d'%(n_pool+1))
#         stack = DenseBlock(stack, n_layers_per_block[n_pool], growth_rate, dropout_p, scope='denseblockbn%d'%(n_pool+1))
        
#         # Повышаем дискретизацию #
#         for i in range(n_pool):
#             # Переход выше (повышение дискр. + сложение с skip_connection)
#             n_filters_keep = growth_rate * n_layers_per_block[n_pool + 1]
# #             stack = TransitionUp(block_to_upsample, skip_connection_list[i], n_filters_keep, scope='transitionup%d' % (n_pool + i + 1))
#             stack = TransitionUp(stack, 1, n_filters_keep, scope='transitionup%d' % (n_pool + i + 1))
            
#             # Мы будем поднимать только новые карты признаков
# #             stack, block_to_upsample = DenseBlock(stack, n_layers_per_block[n_pool + i + 1], growth_rate, dropout_p, scope='denseblock%d' % (n_pool + i + 2))
#             stack = DenseBlock(stack, n_layers_per_block[n_pool + i + 1], growth_rate, dropout_p, scope='denseblock%d' % (n_pool + i + 2))
        
#         # Финальный Softmax #
#         output = l.Conv2D(num_classes, (1, 1), activation='softmax', name='output')(stack)
#         model = tfk.models.Model(inputs, output)
# #         model.summary()
#         return model


import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as l



def dense_block(input_tensor, block_num, num_layers, growth_rate):
    concat_layers = [input_tensor]
    x = input_tensor
    for i in range(num_layers):
        layer = l.BatchNormalization(name=f'bn_{block_num}_{i}')(x)
        layer = l.ReLU(name=f'relu_{block_num}_{i}')(layer)
        layer = l.Conv2D(growth_rate, (3, 3), padding='same', name=f'conv_{block_num}_{i}')(layer)
        concat_layers.append(layer)
        x = l.concatenate(concat_layers, axis=-1, name=f'concat_{block_num}_{i}')
    return x

def transition_down(input_tensor, block_num):
    x = l.BatchNormalization(name=f'bn_td_{block_num}')(input_tensor)
    x = l.ReLU(name=f'relu_td_{block_num}')(x)
    x = l.Conv2D(int(x.shape[-1]) // 2, (1, 1), name=f'conv_td_{block_num}')(x)
    x = l.MaxPooling2D((2, 2), strides=(2, 2), name=f'maxpool_td_{block_num}')(x)
    return x

def build_tiramisu(input_shape, num_classes, num_blocks=5, num_layers_per_block=4, growth_rate=16):
    input_layer = l.Input(shape=input_shape, name='input')
    x = input_layer
    
    skip_connections = []

    for i in range(num_blocks):
        x = dense_block(x, i+1, num_layers_per_block, growth_rate)
        skip_connections.append(x)
        if i != num_blocks - 1:
            x = transition_down(x, i+1)

    x = dense_block(x, num_blocks+1, num_layers_per_block, growth_rate)

    for i in range(num_blocks - 1, -1, -1):
        x = l.Conv2DTranspose(int(skip_connections[i].shape[-1]) // 2, (3, 3), strides=(2, 2), padding='same')(x)
        x = tf.image.resize(x, skip_connections[i].shape[1:3])
        x = l.concatenate([x, skip_connections[i]], axis=-1)
        x = dense_block(x, num_blocks+1+i, num_layers_per_block, growth_rate)

    output_layer = l.Conv2D(num_classes, (1, 1), activation='softmax', name='output')(x)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    return model