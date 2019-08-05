import torch
import cv2
from collections import deque
import seaborn as sns; sns.set(style="ticks", rc={"lines.linewidth": 0.5})
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg',warn=False, force=True)
import matplotlib.pyplot as plt
print(sns.__version__)

CUBOIDS = 10


class ModelTester:
    IMG_WIDTH = 227
    IMG_HT = 227
    SEQUENCE_STRIDES = [i for i in range(0, CUBOIDS)]

    def __init__(self, model, test_videos_file):
        self.model = model
        self.test_videos_file = test_videos_file
        self.frame_buffer = deque([], maxlen=len(self.SEQUENCE_STRIDES))
        # allocate memory for cuboids
        self.cuboids = np.zeros((len(self.SEQUENCE_STRIDES),
                                 self.IMG_HT, self.IMG_WIDTH), dtype=np.float16)

    def test(self):
        video = cv2.VideoCapture(self.test_videos_file)
        errors = []
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            dim = (self.IMG_WIDTH, self.IMG_HT)
            frame = cv2.resize(frame, dim)
            frame = frame.astype('float16')
            # Normalizing the value
            frame = frame / 255  # - self.mean
            self.frame_buffer.append(frame)

            if len(self.frame_buffer) < len(self.SEQUENCE_STRIDES):
                continue

            for index, frame_no in enumerate(self.SEQUENCE_STRIDES):
                self.cuboids[index] = self.frame_buffer[frame_no]

            cuboid = torch.from_numpy(self.cuboids).double()
            cuboids = cuboid.unsqueeze(0)

            output = self.model(cuboids)

            output_numpy = output.cpu().detach().numpy()[0]
            for index in range(0, len(self.SEQUENCE_STRIDES)):
                reconstructed_image = output_numpy[index]
                error = np.abs(np.subtract(self.cuboids[index], reconstructed_image))
                errors.append(np.sum(error))
                cv2.imwrite("test1.png", reconstructed_image * 255)
                cv2.imwrite("test2_1.png", cuboids.cpu().detach().numpy()[0][0] * 255)

            self.frame_buffer.clear()
        min_error = min(errors)
        max_error = max(errors)
        regularity_scores = []
        for index, error in enumerate(errors):
            regularity_scores.append(1 - ((error - min_error) / max_error))

        episodes = list(range(0, len(errors)))
        data_preproc = pd.DataFrame({
            'Episodes': episodes,
            'score': regularity_scores})

        sns.lineplot(x='Episodes', y='value', hue='variable',
                     data=pd.melt(data_preproc, ['Episodes']))
        plt.show()

