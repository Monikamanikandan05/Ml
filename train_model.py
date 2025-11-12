# -----------------------------------------------------------
# Model Training Script for Traffic Sign Recognition
# -----------------------------------------------------------
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_dir = 'dataset/train'
test_dir = 'dataset/test'

datagen = ImageDataGenerator(rescale=1./255, rotation_range=10, zoom_range=0.1)
train_gen = datagen.flow_from_directory(train_dir, target_size=(32, 32), batch_size=32, class_mode='categorical')
test_gen = datagen.flow_from_directory(test_dir, target_size=(32, 32), batch_size=32, class_mode='categorical')

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_gen.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_gen, epochs=10, validation_data=test_gen)

model.save('traffic_model.h5')
print("✅ Model saved as traffic_model.h5")
