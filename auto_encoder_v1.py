import os
from os import listdir
from os.path import isfile, join

import datetime
import cv2
import torch
from torch import nn
import numpy as np
import torch.optim as optim

from torch.optim.adagrad import Adagrad
import torch.nn.functional as F

from torch.utils.data.dataloader import DataLoader
from training_data_loader import TrainingFileManager, TrainingDataSampler, RegularizationDataset


# This can be used to load all the training data into the memory
PARENT_DIR = "./enter"

# This can be used to load large datasets and excerise caching as in TrainingFileManager class
PARENT_DIR_LIST = ['./enter']


INPUT_DATA_PATH = "{}/{}".format(PARENT_DIR, "input_data")
MODEL_PATH = "{}/{}".format(PARENT_DIR, "model")
LOSSES_FILE_PATH = "{}/{}".format(MODEL_PATH, "losses.npy")
MODEL_FILE_PATH = "{}/{}".format(MODEL_PATH, "trained_model")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if not os.path.exists(INPUT_DATA_PATH):
    os.makedirs(INPUT_DATA_PATH)

if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

# Total number of episodes to train the Network
EPISODES_COUNT = 20000
# This mode will train the model. In load, mode, model.bak.bak.bak.bak can be tested.
RUN_MODE = "save"
# Number of episodes after which model should be saved periodically
SNAPSHOT_DURATION = 10000
# Load the entire training dataset in memory
LOAD_ALL_MEMORY = False


class AutoEncoderModel(nn.Module):
    def __init__(self):
        super(AutoEncoderModel, self).__init__()

        # This takes in 10 X 227 X 277 and spits out 512 X 55 X 55
        self.encoder_conv1 = nn.Conv2d(in_channels=10, out_channels=512, kernel_size=11, stride=4)
        nn.init.xavier_uniform_(self.encoder_conv1.weight)

        # This takes in 512 X 55 X 55 and spits out 512 X 27 X 27
        self.encoder_pool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        # This takes in 512 X 27 X 27 and spits out  256 X 27 X 27
        self.encoder_conv2 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=5, stride=1,
                                       padding=2)
        nn.init.xavier_uniform_(self.encoder_conv2.weight)

        # This takes in 256 X 27 X 27 and spits out 256 X 13 X 13
        self.encoder_pool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        # This takes in 256 X 13 X 13 and spits out 128 X 13 X 13
        self.encoder_conv3 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1,
                                       padding=1)
        nn.init.xavier_uniform_(self.encoder_conv3.weight)

        # This takes in 128 X 13 X 13 and spits out 256 X 13 X 13 and is the first deconv layer
        self.decoder_conv1 = nn.ConvTranspose2d(in_channels=128, out_channels=256, kernel_size=3,
                                                padding=1)
        nn.init.xavier_uniform_(self.decoder_conv1.weight)

        # This takes in 256 X 13 X 13 and spits out 256 X 27 X 27
        self.decoder_unpool1 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.decoder_unpool1_size = (27, 27)

        # This takes in 256 X 27 X 27 and spits out 512 X 27 X 27
        self.decoder_conv2 = nn.ConvTranspose2d(in_channels=256, out_channels=512, kernel_size=5,
                                                stride=1, padding=2)
        nn.init.xavier_uniform_(self.decoder_conv2.weight)

        # This takes in 512 X 27 X 27 and spits out 512 X 55 X 55
        self.decoder_unpool2 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.decoder_unpool2_size = (55, 55)

        # This takes in 512 X 55 X 55 and spits out 10 X 227 X 227
        self.decoder_output = nn.ConvTranspose2d(in_channels=512, out_channels=10, kernel_size=11,
                                                 stride=4)
        nn.init.xavier_uniform_(self.decoder_output.weight)

    def forward(self, input_data):

        # Encoding logic

        # The first convolution layer takes in 10 X 227 X 277 and spits out 512 X 55 X 55
        encoder_conv1_output = F.tanh(self.encoder_conv1(input_data.float()))

        # The first pooling layer takes in 512 X 55 X 55 and spits out 512 X 27 X 27
        encoder_pool1_output, pool1_indices = self.encoder_pool1(encoder_conv1_output)

        # The second convolution layer takes in 512 X 27 X 27 and spits out  256 X 27 X 27
        encoder_conv2_output = F.tanh(self.encoder_conv2(encoder_pool1_output))

        # The second pooling layer takes in 256 X 27 X 27 and spits out 256 X 13 X 13
        encoder_pool2_output, pool2_indices = self.encoder_pool2(encoder_conv2_output)

        # The third convolution layer 256 X 13 X 13 and spits out 128 X 13 X 13
        encoder_conv3_output = F.tanh(self.encoder_conv3(encoder_pool2_output))

        # Decoding logic

        # The first deconvolution layer takes in 128 X 13 X 13 and spits out 256 X 13 X 13
        decoder_conv1_output = F.tanh(self.decoder_conv1(encoder_conv3_output))

        # First unpool layer takes in 256 X 13 X 13 and spits out 256 X 27 X 27
        decoder_unpool1_output = self.decoder_unpool1(decoder_conv1_output,
                                                      pool2_indices,
                                                      output_size=self.decoder_unpool1_size)

        # Second deconvolution layer takes in 256 X 27 X 27 and spits out 512 X 27 X 27
        decoder_conv2_output = F.tanh(self.decoder_conv2(decoder_unpool1_output))

        # Second unpooling layer takes in  512 X 27 X 27 and spits out 512 X 55 X 55
        decoder_unpool2_output = self.decoder_unpool2(decoder_conv2_output,
                                                      pool1_indices,
                                                      output_size=self.decoder_unpool2_size)

        # The third deconvolution layer takes in 512 X 55 X 55 and spits out 10 X 227 X 227
        decoder_output = F.tanh(self.decoder_output(decoder_unpool2_output))
        return decoder_output


class AutoEncoder:

    BATCH_SIZE = 32

    def __init__(self, model):
        # if torch.cuda.device_count() > 1:
        #    self.model.bak.bak.bak.bak = nn.DataParallel(model.bak.bak.bak.bak)
        # else:
        self.model = model

        # Default is the same model.bak.bak.bak.bak
        self.optimizer = None
        self.scheduler = None
        self.criterion = nn.MSELoss()
        self.losses = []

    def post_setup(self):
        self.optimizer = Adagrad(self.model.parameters(), lr=0.001, weight_decay=0.0005)

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.5,
                                                              patience=20,
                                                              threshold=0.0001)

    def train_batches(self, episodes_count, files_list):
        episode_counter = 0
        prev_loss = 0.0

        if LOAD_ALL_MEMORY:
            dataset_loader = TrainingDataSampler(EPISODES_COUNT)
            dataset_loader.load_all_training_data(files_list)
        else:
            dataset_loader = TrainingFileManager(PARENT_DIR_LIST, EPISODES_COUNT)

        dataset_loader.start()

        while episode_counter < episodes_count:

            cuboids = dataset_loader.get_training_data()

            print("{} : Running episode {} and prev loss {}".format(datetime.datetime.now(),
                                                                    str(episode_counter),
                                                                    prev_loss))
            cuboids = cuboids.to(DEVICE)
            output = self.model(cuboids)
            output = output.to(DEVICE)
            self.optimizer.zero_grad()  # zero the gradient buffers
            loss = self.criterion(output, cuboids)
            loss.backward()
            self.optimizer.step()  # Does the update
            self.losses.append(loss.item())

            if episode_counter > 0 and episode_counter % SNAPSHOT_DURATION == 0:
                np.save("{}-{}".format(LOSSES_FILE_PATH, str(episode_counter)), np.array(self.losses))
                torch.save(auto_encoder, "{}-{}".format(MODEL_FILE_PATH, str(episode_counter)))
                cv2.imwrite("test1.png", cuboids.cpu().detach().numpy()[0][0] * 255)
                cv2.imwrite("test2_1.png", output.cpu().detach().numpy()[0][0] * 255)

            print("Loss for episode {} is {}".format(episode_counter, loss))
            prev_loss = loss.item()
            episode_counter += 1

        np.save("{}-{}".format(LOSSES_FILE_PATH, str(episode_counter)), np.array(self.losses))
        torch.save(auto_encoder, "{}-{}".format(MODEL_FILE_PATH, str(episode_counter)))


if __name__ == "__main__":
    if RUN_MODE == "save":

        auto_encoder_model = AutoEncoderModel()
        auto_encoder_model = auto_encoder_model.float()
        auto_encoder_model.to(DEVICE)

        auto_encoder = AutoEncoder(auto_encoder_model)
        auto_encoder.post_setup()

        files = []
        for file in listdir(INPUT_DATA_PATH):
            full_file_path = join(INPUT_DATA_PATH, file)
            if isfile(full_file_path):
                files.append(full_file_path)

        auto_encoder.train_batches(EPISODES_COUNT, np.array(files))
    else:
        losses = np.load(LOSSES_FILE_PATH)
        print(losses)
