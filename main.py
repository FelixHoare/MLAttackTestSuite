import vgg
import utkface_loader
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

UTKFACE_PATH = "/Users/felixhoare/ComputerScience/YearFour/Diss/Data/utkcropped"

print("Processing images...")

dataframe = utkface_loader.parse_utkface_data(UTKFACE_PATH)

print(dataframe)

x, y = [], []

for i in range(len(dataframe)):
    dataframe['images'].iloc[i] = dataframe['images'].iloc[i].resize((200, 200))
    array = np.asarray(dataframe['images'].iloc[i])
    x.append(array)
    #agegenrace = [int(dataframe['ages'].iloc[i]), int(dataframe['genders'].iloc[i], int(dataframe['races'].iloc[i]))]
    #y.append(agegenrace)

x = np.array(x)

y_race = tf.keras.utils.to_categorical(dataframe['races'], num_classes=5)

print("Splitting data...")

x_train, x_test, y_train_race, y_test_race = train_test_split(x, y_race, test_size=0.2, random_state=42)

print("Creating model...")

model = vgg.create_vgg_ll()

print("Fitting model...")

history = model.fit(x_train, y_train_race, epochs=12, batch_size=32, validation_data=(x_test, y_test_race))

print("Evaluating model...")

model.evaluate(x_test, y_test_race)

