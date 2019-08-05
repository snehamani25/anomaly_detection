import os
from os import listdir
from os.path import isfile, join

import cv2
from collections import deque
import numpy as np
import ntpath

PARENT_DIR = ["enter"]
OUT_DIR = "./enter"
TRAINING_VIDEOS = "training_videos"
MODEL_PATH = "model"
INPUT_DATA_PATH = "{}/{}".format(OUT_DIR, "input_data")

if not os.path.exists(INPUT_DATA_PATH):
    os.makedirs(INPUT_DATA_PATH)


class FeatureExtractor:
    BATCH_SIZE = 1
    IMG_WIDTH = 227
    IMG_HT = 227
    SEQUENCE_NUMBER = 0
    WINDOW_LENGTH = 10
    STRIDE = 2  # This stride is used for sampling the cuboids

    # This is the strides used for sampling the frames
    SEQUENCE_STRIDES = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                        [0,2,4,6,8,10,12,14,16,18],
                        [0,3,6,9,12,15,18,21,24,27],
                        [0,4,8,12,16,20,24,28,32,36]]

    def __init__(self,video_file_path, window_length, stride_index, parent):
        self.video_file_path = video_file_path
        self.window_length = window_length
        self.stride_index = self.SEQUENCE_NUMBER
        self.parent = parent

        # process frames for the video
        self.frame_buffer = deque([], maxlen=window_length)

        # allocate memory for cuboids
        self.cuboids = np.zeros((self.BATCH_SIZE, len(self.SEQUENCE_STRIDES[stride_index]),
                                 self.IMG_HT, self.IMG_WIDTH),
                                dtype=np.float16)

        # Count of number of batches for a stride
        self.count = 0

        # Batch count tracker
        self.batch_no = 0

        self.frame_counter = 0

    def _get_file_name_for_batch(self, video_file_path, stride_index, batch_no, cuboids_count):
        return "{}_{}_seq-{}_batch-{}_stride-{}_window-{}_{}".format(
            self.parent, ntpath.basename(video_file_path), str(stride_index+1), str(batch_no),
            self.STRIDE, self.window_length, str(cuboids_count))

    def save_image_frame(self, frame):
        cv2.imwrite('frame_img/frame_img-'+str(self.frame_counter)+".jpg", frame)

    def process_video(self):
        video = cv2.VideoCapture(self.video_file_path)
        # current stride length
        csl = 0

        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            dim = (self.IMG_WIDTH,self.IMG_HT)
            frame = cv2.resize(frame, dim)

            # self.save_image_frame(frame)
            self.frame_counter += 1

            frame = frame.astype('float16')
            # Normalizing the value
            frame = frame / 255 # - self.mean

            # Appending to frame_buffer
            self.frame_buffer.append(frame)

            if len(self.frame_buffer) < self.window_length or csl != 0:
                csl = (csl + 1) % self.STRIDE
                continue

            seq = self.SEQUENCE_STRIDES[self.stride_index]

            for index, frame_no in enumerate(seq):
                self.cuboids[self.count][index] = self.frame_buffer[frame_no]

            csl = (csl + 1) % self.STRIDE

            if self.BATCH_SIZE-1 == self.count:
                filename = self._get_file_name_for_batch(self.video_file_path, self.stride_index,
                                                         self.batch_no, self.count+1)

                np.save("{}/{}".format(INPUT_DATA_PATH, filename), self.cuboids)

                print(self.video_file_path, "batch", self.batch_no, "count", self.count,
                      "frame", self.frame_counter)
                self.batch_no += 1
                self.count = 0
            else:
                self.count += 1

        if self.count > 0:
            filename = self._get_file_name_for_batch(self.video_file_path, self.stride_index,
                                                     self.batch_no, self.count)
            np.save("{}/{}".format(INPUT_DATA_PATH, filename), self.cuboids[0:self.count])
            print(self.video_file_path, "batch", self.batch_no, "count", self.count,
                  "frame", self.frame_counter)


if __name__ == "__main__":

    for parent_dir in PARENT_DIR:
        training_videos_path = "{}/{}".format(parent_dir, TRAINING_VIDEOS)
        for file in listdir(training_videos_path):
            full_file_path = join(training_videos_path, file)
            if isfile(full_file_path):
                feature_ext = FeatureExtractor(full_file_path,
                                               FeatureExtractor.WINDOW_LENGTH,
                                               FeatureExtractor.SEQUENCE_NUMBER,
                                               parent_dir)
                feature_ext.process_video()
