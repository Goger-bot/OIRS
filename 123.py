import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

# размеры изображений
img_width, img_height = 150, 150

# пути к тренировочному и валидационному датасетам
train_data_dir = 'train'
validation_data_dir = 'val'

# количество эпох
epochs = 8

# количество классов (гепард или лев)
num_classes = 2

# генератор тренировочных данных
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=15,  # Уменьшено с 40
    width_shift_range=0.2,  # Уменьшено с 0.2
    height_shift_range=0.2,  # Уменьшено с 0.2В
    zoom_range=0.1,  # Уменьшено с 0.2
    horizontal_flip=True
)

# генератор валидационных данных
validation_datagen = ImageDataGenerator(rescale=1./255)

# генератор тренировочного датасета
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='binary',
    subset='training',
    shuffle=True)

# генератор валидационного датасета
validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='binary',
    shuffle=False)

# создание модели
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(img_width, img_height, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# компиляция модели
model.compile(loss='binary_crossentropy',
    optimizer='rmsprop',
    metrics=['accuracy'])

# вычисление количества шагов на эпоху для валидационного генератора
val_steps = len(validation_generator.filenames) // validation_generator.batch_size

# проверка, содержит ли валидационный генератор изображения
val_images, val_labels = next(validation_generator)
print(val_images.shape, val_labels.shape)

# если валидационный генератор не содержит изображений, то прерываем выполнение кода
if val_images.shape[0] == 0:
    print('Validation generator does not contain any images')
    exit()
drop_remainder=True

# обучение модели
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=val_steps)

# оценка модели
score = model.evaluate(validation_generator, steps=val_steps)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# построение графиков точности и потерь
import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

fig, axs = plt.subplots(1, 2, figsize=(12, 6))
fig.suptitle('Training and Validation Accuracy and Loss', fontsize=16)

axs[0].plot(epochs_range, acc, label='Training Accuracy')
axs[0].plot(epochs_range, val_acc, label='Validation Accuracy')
axs[0].set_ylim([0, 1])
axs[0].set_xlabel('Epochs')
axs[0].set_ylabel('Accuracy')
axs[0].legend(loc='lower right')

axs[1].plot(epochs_range, loss, label='Training Loss')
axs[1].plot(epochs_range, val_loss, label='Validation Loss')
axs[1].set_ylim([0, 1])
axs[1].set_xlabel('Epochs')
axs[1].set_ylabel('Loss')
axs[1].legend(loc='upper right')

plt.show()

# Сохранение модели
model.save('model.h5')

# Загрузка модели (для проверки)
from tensorflow.keras.models import load_model
loaded_model = load_model('model.h5')