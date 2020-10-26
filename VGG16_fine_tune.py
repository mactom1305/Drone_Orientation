from keras.applications import VGG16
from keras import models
from keras import layers
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from keras.models import load_model
from keras import callbacks

# Settings
train_dir = 'Data_sets/Data_set_3000_17.20.31_224x224'
# Change the batchsize according to your system RAM
train_batchsize = 100
val_batchsize = 10

# Prepare data generator
datagen = ImageDataGenerator(validation_split=0.1)



train_generator = datagen.flow_from_directory(
    train_dir,
    subset='training',
    target_size=(image_size, image_size),
    batch_size=train_batchsize,
    class_mode='categorical')

validation_generator = datagen.flow_from_directory(
    train_dir,
    subset='validation',
    target_size=(image_size, image_size),
    batch_size=val_batchsize,
    class_mode='categorical',
    shuffle=False)

# #Load the VGG model
image_size = 224
vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))

# Freeze the layers except the last 4 layers
for layer in vgg_conv.layers[:-4]:
    layer.trainable = False

# Check the trainable status of the individual layers
for layer in vgg_conv.layers:
    print(layer, layer.trainable)

# Create the model
model = models.Sequential()
# Add the vgg convolutional base model
model.add(vgg_conv)

# Add new layers
model.add(layers.Flatten())
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(3000, activation='softmax'))

# Show a summary of the model. Check the number of trainable parameters
model.summary()



# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

image_size = 224
model = load_model('Trained_networks/vgg16_tuned_26lut.h5')
model.summary()

# Train the model

model_checkpoint = callbacks.ModelCheckpoint('Trained_networks/vgg16_tuned{epoch:02d}.h5',period=2,save_weights_only=False)
callbacks.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
history = model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.samples / train_generator.batch_size,
    epochs=90,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples / validation_generator.batch_size,
    verbose=1)

# Save the model
model.save('Trained_networks/vgg16_tuned_27lut.h5')

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
