from random import seed
from time import sleep
import numpy as np
import cv2
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from keras import layers
from keras.constraints import maxnorm
from keras.utils import np_utils
from imutils import paths
import matplotlib.pyplot as plt

def plotGraphs(history, epochs: int)->None:
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(len(acc))

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

def conv_2d_pooling_layers(filters, number_colour_layers: int):
    return [
        tf.keras.layers.Conv2D(
            filters,
            number_colour_layers,
            padding='same',
            activation='relu'
        ),
        tf.keras.layers.MaxPooling2D()
    ]


def main():
    color_mode = 'RGB'
    number_colour_layers = 3
    image_size = (200, 200)
    image_shape = image_size + (number_colour_layers,)

    #paths das imagens
    training_data_path = './Cartas/data/train'
    test_data_path = './Cartas/data/test'
    val_data_path = './Cartas/data/validation'

    #armazenar os ds
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_data_path,
        seed=42,
        image_size=image_size,
        batch_size=32
    )
    train_ds = tf.keras.utils.image_dataset_from_directory(
        training_data_path,
        seed=42,
        image_size=image_size,
        batch_size=32
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_data_path,
        seed=42,
        image_size=image_size,
        batch_size=32
    )


    class_names = train_ds.class_names

    #print(class_names)    
    # plt.figure(figsize=(15, 15))
    # for images, labels in train_ds.take(1):
    #     for i in range(20):
    #         ax = plt.subplot(5, 5, i + 1)
    #         plt.imshow(images[i].numpy().astype("uint8"))
    #         plt.title(class_names[labels[i]])
    #         plt.axis("off")
    # plt.show()

    #Config for performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
    num_classes = len(class_names)

    preprocessing_layers = [
        tf.keras.layers.experimental.preprocessing.Rescaling(1./255, input_shape=image_shape)
    ]

    core_layers = \
        conv_2d_pooling_layers(16, number_colour_layers) + \
        conv_2d_pooling_layers(32, number_colour_layers) + \
        conv_2d_pooling_layers(64, number_colour_layers)

    dense_layers = [
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation=tf.nn.softmax)
    ]

    #juntar os layers com o modelo sequencial
    model = tf.keras.Sequential(
        preprocessing_layers +
        core_layers +
        dense_layers
    )

    #loss usando categorial do keras
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    #compilar o modelo usando o otimizador adam
    model.compile(
        optimizer='adam',
        loss=loss,
        metrics=['accuracy']
    )

    #Callback para evitar overfitting
    callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto',
        baseline=None, restore_best_weights=True
    )
    
    #Fit do modelo, usando o ds de treino e de validação
    history = model.fit(
        train_ds,
        validation_data = val_ds,
        epochs = 10,
        callbacks = [callback]
    )
    
    #acuracia com o ds de teste
    print(model.evaluate(test_ds))
    #plot de gráficos
    plotGraphs(history=history, epochs=10)

    CardPath = './Testes/9.jpg'
    Card = tf.keras.utils.load_img(
        CardPath, target_size=image_size
    )
    Card_array = tf.keras.utils.img_to_array(Card)
    Card_array = tf.expand_dims(Card_array, 0) # Create a batch
    predictions = model.predict(Card_array)
    score = tf.nn.softmax(predictions[0])
    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )

if __name__ == "__main__":
    main()