import os
import sys
import argparse
import matplotlib.pyplot as plt

from keras import __version__
# from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.applications.inception_v3 import InceptionV3, preprocess_input

from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD

print('import done')


IM_WIDTH,IM_HEIGHT = 299 , 299
FC_SIZE = 1024
NB_IV3_LAYERS_TO_FREEZE = 172
batch_size = 16
train_dir = 'data/train_sym'
val_dir = 'data/validation_sym'

train_datagen = ImageDataGenerator(
                preprocessing_function=preprocess_input,
                rotation_range=30,
                width_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True)


test_datagen = ImageDataGenerator(
                preprocessing_function=preprocess_input,
                rotation_range=30,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True)
# 训练数据与测试数据

train_generator = train_datagen.flow_from_directory(
                                                    train_dir,
                                                    target_size=(IM_WIDTH,IM_HEIGHT),
                                                    batch_size=batch_size,
                                                    class_mode='categorical')
validation_generator = test_datagen.flow_from_directory(
                                                    val_dir,
                                                    target_size=(IM_WIDTH,IM_HEIGHT),
                                                    batch_size=batch_size,
                                                    class_mode='categorical')

#添加新层
def add_new_last_layer(base_model,nb_classes):
    '''
      添加最后的层
      输入:base_model和分类数目
      输出:新的keras的model
    '''
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(FC_SIZE,activation='relu')(x)#new FC layer,random init
    predictions = Dense(nb_classes,activation='softmax')(x)
    model = Model(input=base_model.input,output=predictions)
    
    return model

#冻上base_model所有层,这样就可以正确获得bottleneck特征

def setup_to_transfer_learn(model,base_model):
    '''freeze all layers and compile '''
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer='rmsprop',loss='categroical_crossentropy',metrics=['accuracy'])
    
#定义网络框架
nb_classes = 5
base_model = InceptionV3(weights='imagenet',include_top=False)
model = add_new_last_layer(base_model,nb_classes)
setup_to_transfer_learn(model,base_model)

#模式--训练
histroy_tl = model.fit_generator(
            train_generator,
            nb_epoch=3,
            samples_per_epoch=400//batch_size,
            validation_data=validation_generator,
            nb_val_samples=100//batch_size,
            class_weight='auto')
    
    
    
