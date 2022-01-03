# Arda Mavi

import os.path
from keras.models import Sequential
from keras.models import model_from_json

def predict(model, X):
    Y = model.predict(X)
    Y = np.argmax(Y, axis=1)
    Y = 'cat' if Y[0] == 0 else 'dog'
    return Y

if __name__ == '__main__':
    import sys
    img_dir = sys.argv[1]
    if not os.path.exists('output'):
        os.mkdir('output')
    from get_dataset import get_img
    img = get_img(img_dir)
    import numpy as np
    X = np.zeros((1, 64, 64, 3), dtype='float64')
    X[0] = img
    # Getting model:
    model_file = open('Data/Model/model.json', 'r')
    model = model_file.read()
    model_file.close()
    model = model_from_json(model)
    # Getting weights
    model.load_weights("Data/Model/weights.h5")
    Y = predict(model, X)
    # append result to file
    with open('output/task1.txt', 'a') as f:
        f.write(img_dir + ': ' + Y + '\n')
    print('It is a ' + Y + ' !')
