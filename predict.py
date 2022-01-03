# Arda Mavi
import os.path
import cv2
import numpy as np
from scipy.misc import imresize
import time

from keras.models import Sequential
from keras.models import model_from_json

def predict(model, X):
    Y = model.predict(X)
    Y = np.argmax(Y, axis=1)
    Y = 'cat' if Y[0] == 0 else 'dog'
    return Y

def check_frame(name, img):
    X = np.zeros((1, 64, 64, 3), dtype='float64')
    X[0] = imresize(img, (64, 64, 3))
    # Getting model:
    model_file = open('Data/Model/model.json', 'r')
    model = model_file.read()
    model_file.close()
    model = model_from_json(model)
    # Getting weights
    model.load_weights("Data/Model/weights.h5")
    Y = predict(model, X)
    return Y

if __name__ == '__main__':
    import sys
    img_dir = sys.argv[1]
    if not os.path.exists('output'):
        os.mkdir('output')
    if not os.path.exists('output/cats'):
        os.mkdir('output/cats')
    if not os.path.exists('output/dogs'):
        os.mkdir('output/dogs')
    filename, file_extension = os.path.splitext(img_dir)
    frame_count = 1
    if file_extension.__eq__('.mp4'):
        cap = cv2.VideoCapture(img_dir)
        while cap.isOpened():
            ret, frame = cap.read()
            try:
                cv2.imshow('frame', frame)
                Y = check_frame(img_dir, frame)
                # append result to file
                with open('output/task2.txt', 'a') as f:
                    f.write(img_dir + str(frame_count) + ': ' + Y + '\n')
                print('It is a ' + Y + ' !')
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                # save compressed images in cats and dogs folders:
                try:
                    if Y.__eq__('cat'):
                        cv2.VideoWriter('output/cats/video_frame' + str(frame_count) + '.png',
                                    frame, [cv2.IMWRITE_PNG_COMPRESSION, 9])
                    else:
                        cv2.imwrite('output/dogs/video_frame' + str(frame_count) + '.png',
                                    frame, [cv2.IMWRITE_PNG_COMPRESSION, 9])
                except:
                    cv2.imwrite('output/dogs/video_frame' + str(frame_count) + '.jpg', frame)
                time.sleep(10)
            except:
                print('frame ' + str(frame_count) + 'was not readable')
                break
            frame_count = frame_count + 1
        cap.release()
        cv2.destroyAllWindows()
    else:
        from get_dataset import get_img
        img = get_img(img_dir)
        Y = check_frame(img_dir, img)
        # append result to file
        with open('output/task2.txt', 'a') as f:
            f.write(img_dir + ': ' + Y + '\n')
        print('It is a ' + Y + ' !')

