# Arda Mavi
import sys
import os.path
import cv2
import queue
import numpy as np
from scipy.misc import imresize
import time
import threading
from get_dataset import get_img

from keras.models import Sequential
from keras.models import model_from_json


class Prediction:
    def __init__(self):
        # Getting model:
        model_file = open('Data/Model/model.json', 'r')
        model = model_file.read()
        model_file.close()
        self.model = model_from_json(model)
        # Getting weights
        self.model.load_weights("Data/Model/weights.h5")

    def predict(self, X):
        Y = self.model.predict(X)
        Y = np.argmax(Y, axis=1)
        Y = 'cat' if Y[0] == 0 else 'dog'
        return Y

    def check_frame(self, img):
        X = np.zeros((1, 64, 64, 3), dtype='float64')
        X[0] = imresize(img, (64, 64, 3))
        Y = self.predict(X)
        return Y


class CloudStorage:

    def __init__(self, address, filename):
        # Simulate connection to cloud 'output/cats'
        if not os.path.exists(address):
            os.mkdir(address)
        self.to_send = queue.Queue()
        self.address = address
        self.cap = cv2.VideoCapture(filename)
        self.frame_count = 0
        x = threading.Thread(target=self.uploading)
        x.start()

    # running on thread, runs on queue containing frame indexes to upload. index -1 at the end
    def uploading(self):
        ret = False
        frame = None
        while True:
            if self.to_send.empty():
                time.sleep(1)
                continue
            # we have a frame to send
            frame_to_send = self.to_send.get()
            if frame_to_send == -1:
                # no more frames to send
                break
            # continue reading frames until frame number == 'frame_to_send'
            while self.frame_count < frame_to_send:
                ret, frame = self.cap.read()
                self.frame_count = self.frame_count + 1
            # upload the frame
            if ret:
                self.upload_frame(frame, self.frame_count)
        # no more frame to upload. release thread
        self.cap.release()


    def add_frame(self, index):
        self.to_send.put(index)

    # upload_frame serves as a simulator to uploading
    def upload_frame(self, frame, frame_count):
        time.sleep(1)
        try:
            # save compressed images in cats and dogs folders:
            destination = '{}/video_frame{}.png'.format(self.address, frame_count)
            cv2.imwrite(destination, frame, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        except:
            print('Failed to save compressed image to ' + destination)

def check_input(img_dir):
    # send full path, file_name and weather it is a video
    path_name = os.path.split(img_dir)
    if not path_name[1]:
        print('no file entered')
        exit(1)
    filename, file_extension = os.path.splitext(path_name[1])
    is_video = file_extension.__eq__('.mp4')
    return img_dir, filename, is_video

if __name__ == '__main__':
    path, filename, is_video = check_input(sys.argv[1])
    predict_cls = Prediction()
    if not os.path.exists('output'):
        os.mkdir('output')
    f = open('output/task2.txt', 'a')
    # leave the option of checking a single frame:
    if not is_video:
        img = get_img(path)
        res = predict_cls.check_frame(img)
        # append result to file
        f.write(filename + ': ' + res + '\n')
        print('It is a ' + res + ' !')
        exit(0)
    # for video:
    # create instances of storage-upload for each classification:
    cat_storage = CloudStorage('output/cats', path)
    dog_storage = CloudStorage('output/dogs', path)
    frame_count = 1
    cap = cv2.VideoCapture(path)
    # read all frames from video
    while cap.isOpened():
        try:
            ret, frame = cap.read()
            if not ret:
                # end of video
                break
            res = predict_cls.check_frame(frame)
            # append result to file
            f.write(filename + str(frame_count) + ': ' + res + '\n')
            print('Frame {} is a {} !'.format(str(frame_count), res))
            # send the frame index to the queue of the relevant instance (dog or cat)
            if res.__eq__('cat'):
                cat_storage.add_frame(frame_count)
            else:
                dog_storage.add_frame(frame_count)
        except:
            print('frame ' + str(frame_count) + ' was not readable')
        frame_count = frame_count + 1
    # indicate end of frames in queue
    cat_storage.add_frame(-1)
    dog_storage.add_frame(-1)
    cap.release()
    f.close()


