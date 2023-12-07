import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
import matplotlib.pyplot as plt

print("GPU disponibili:", tf.config.experimental.list_physical_devices('GPU'))
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

'''
Training directory and global variables
'''
images_repo = "images" #directory containing the images dataset
image_size = (128,128)
batch_size = 16
'''
Method to load dataset
'''
def prepare_dataset():
    train_set = tf.keras.preprocessing.image_dataset_from_directory(images_repo, image_size=image_size, color_mode='rgb', labels='inferred', label_mode='int', shuffle=True, seed=1, validation_split=0.2, interpolation='lanczos5', subset="training", batch_size=batch_size)
    test_set = tf.keras.preprocessing.image_dataset_from_directory(images_repo, image_size=image_size, color_mode='rgb', labels='inferred', label_mode='int', shuffle=True, seed=1, validation_split=0.2, interpolation='lanczos5', subset="validation", batch_size=batch_size)

    train_set = train_set.map(lambda x, y: (tf.image.convert_image_dtype(x, tf.float32), y))
    test_set = test_set.map(lambda x, y: (tf.image.convert_image_dtype(x, tf.float32), y))

    return train_set, test_set

def create_model():

    input_size = (128, 128, 3)

    inputs = Input(input_size)
    conv0 = Conv2D(64, 3, padding='same')(inputs)
    conv0 = LeakyReLU()(conv0)
    conv0 = BatchNormalization()(conv0)
    pool0 = MaxPooling2D(pool_size=(2, 2))(conv0)
    pool0 = Dropout(0.4)(pool0)

    conv1 = Conv2D(512, 3, padding='same')(pool0)
    conv1 = LeakyReLU()(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool1 = Dropout(0.4)(pool1)

    conv2 = Conv2D(256, 3, padding='same')(pool1)
    conv2 = LeakyReLU()(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool2 = Dropout(0.4)(pool2)

    conv3 = Conv2D(128, 3, padding='same')(pool2)
    conv3 = LeakyReLU()(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    pool3 = Dropout(0.4)(pool3)

    conv4 = Conv2D(64, 3, padding='same')(pool3)
    conv4 = LeakyReLU()(conv4)
    conv4 = BatchNormalization()(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    pool4 = Dropout(0.5)(pool4)

    dense = Flatten()(pool4)
    output = Dense(1,activation='sigmoid')(dense)

    model = Model(inputs=inputs, outputs=output)
    model.summary()
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

    return model

training_set, test_set = prepare_dataset()

# dividing training set in training (70%) and validation (10%) set
total_samples = tf.data.experimental.cardinality(training_set).numpy() + tf.data.experimental.cardinality(test_set).numpy()
n_val_samples = int(0.1 * total_samples)
val_set = training_set.take(n_val_samples)
training_set = training_set.skip(n_val_samples)

model = create_model()

history = model.fit(training_set, validation_data=val_set, epochs=30)
loss, accuracy, precision, recall = model.evaluate(test_set)

f1_score = 2 * (precision * recall) / (precision + recall)
print('Stampo i valori test di accuracy, precision, recall e F1 score:')
print('Accuracy: ' + str(accuracy) + ' Precision: ' + str(precision) + '; Recall: ' + str(recall) + '; F1 score: ' + str(f1_score))

print('Stampo i grafici di accuracy e loss')
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(history.history['precision'])
plt.plot(history.history['val_precision'])
plt.plot(history.history['recall'])
plt.plot(history.history['val_recall'])
plt.title('model precision/recall')
plt.ylabel('precision/recall')
plt.xlabel('epoch')
plt.legend(['train precision', 'val precision','train recall', 'val recall'], loc='upper left')
plt.show()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
