import os
import numpy as np
import tensorflow as tf
import pandas as pd
from PIL import Image

def parse_utkface_data(path):

    print("Running!")

    images, ages, genders, races = [], [], [], []

    for filename in sorted(os.listdir(path)):
        try:
            parts = filename.split('_')
            age = int(parts[0])
            gender = int(parts[1])
            race = int(parts[2])

            if age < 15:
                continue

            ages.append(age)
            genders.append(gender)
            races.append(race)
            images.append(Image.open(path + '/' + filename))

        except Exception as e:
            print(f"Error processing file: {filename} - {e}")
            continue

    images = pd.Series(list(images), name='image')
    ages = pd.Series(list(ages), name='age')
    genders = pd.Series(list(genders), name='gender')
    races = pd.Series(list(races), name='race')

    dataframe = pd.concat([images, ages, genders, races], axis=1)

    return dataframe
