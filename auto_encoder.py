import os
from os import listdir
from os.path import isfile, join

import threading
import logging

import datetime
import cv2
import torch
from torch import nn
import numpy as np
import torch.optim as optim

from torch.optim.adagrad import Adagrad
import torch.nn.functional as F

from training_data_loader import TrainingDataSampler, AllTrainingDatasetLoader, TrainingFileManager
from model_tester import ModelTester


# This can be used to load all the training data into the memory
DATA_SET_TYPE = "avenue"
PARENT_DIR = "./{}".format(DATA_SET_TYPE)

# This can be used to load large datasets and excerise caching as in TrainingFileManager class
PARENT_DIR_LIST = ['./all']


INPUT_DATA_PATH = "{}/{}".format(PARENT_DIR, "input_data")
MODEL_PATH = "{}/{}".format(PARENT_DIR, "model")
LOSSES_FILE_PATH = "{}/{}".format(MODEL_PATH, "losses")
MODEL_FILE_PATH = "{}/{}".format(MODEL_PATH, "trained_model")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LOGS_PATH = "logs"

if not os.path.exists(INPUT_DATA_PATH):
    os.makedirs(INPUT_DATA_PATH)

if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)


# Training parameters
# =========================================
# Total number of episodes to train the Network
EPISODES_COUNT = 40000
# Number of episodes after which model should be saved periodically
SNAPSHOT_DURATION = 10000
# Load the entire training dataset in memory
LOAD_ALL_MEMORY = False
# Number of cuboids while training and testing
CUBOIDS = 10
LEARNING_RATE = 0.001

# Testing parameters
# ==========================================
# This mode will train the model. In load, mode, model.bak.bak.bak.bak can be tested.
RUN_MODE = "train"
# The load path for model
PATH = "{}/model/trained_model-20000".format(DATA_SET_TYPE)


class AutoEncoderModel(nn.Module):
    def __init__(self):
        super(AutoEncoderModel, self).__init__()

        # This takes in 10 X 227 X 277 and spits out 512 X 55 X 55
        self.encoder_conv1 = nn.Conv2d(in_channels=CUBOIDS, out_channels=512, kernel_size=11, stride=4)
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
        self.decoder_output = nn.ConvTranspose2d(in_channels=512, out_channels=CUBOIDS,
                                                 kernel_size=11, stride=4)
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

    def __init__(self, model_name, model, optimizer=None, device=None):
        # if torch.cuda.device_count() > 1:
        #     self.model = nn.DataParallel(model)
        # else:
        self.model = model
        self.model_name = model_name

        self.optimizer = optimizer
        self.scheduler = None
        self.criterion = nn.MSELoss()
        self.losses = []

        fh = logging.FileHandler('{}/{}.log'.format(LOGS_PATH, self.model_name))
        fh.setLevel(logging.DEBUG)
        self.logger = logging.getLogger(self.model_name)
        self.logger.addHandler(fh)
        self.device = DEVICE if not device else device

    def post_setup(self, optimizer=None):
        if optimizer is None:
            print("None opt")
            self.optimizer = Adagrad(self.model.parameters(), lr=LEARNING_RATE, weight_decay=0.0005)
        else:
            print("Opt filled")
            self.optimizer = optimizer
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.5,
                                                              patience=20,
                                                              threshold=0.0001)

    def train_batches(self, episodes_count, training_data_sampler, start_episodes_counter=0):
        episode_counter = start_episodes_counter
        prev_loss = 0.0

        #self.logger.warning("Available keys - %s", training_data_sampler.available_keys)
        #self.logger.warning("Probability map - %s", training_data_sampler.prob_map)

        training_data_sampler.start()

        while episode_counter < episodes_count:

            cuboids = training_data_sampler.get_training_data()

            self.logger.warning("{} : Running episode {} and prev loss {}".
                                format(datetime.datetime.now(), str(episode_counter), prev_loss))
            cuboids = cuboids.to(self.device)
            output = self.model(cuboids)
            output = output.to(self.device)
            self.optimizer.zero_grad()  # zero the gradient buffers
            loss = self.criterion(output, cuboids)
            loss.backward()
            self.optimizer.step()  # Does the update
            self.losses.append(loss.item())

            if episode_counter > 0 and episode_counter % SNAPSHOT_DURATION == 0:
                np.save("{}-{}-{}".format(LOSSES_FILE_PATH, self.model_name, str(episode_counter)),
                        np.array(self.losses))

                torch.save(
                    {
                        'optimizer': self.optimizer.state_dict(),
                        'model': self.model.state_dict()
                    }, "{}-{}-state-{}".format(MODEL_FILE_PATH, self.model_name, episode_counter))

                cv2.imwrite("{}/{}-{}-test1.png".format(LOGS_PATH,
                                                        self.model_name, episode_counter),
                            cuboids.cpu().detach().numpy()[0][0] * 255)
                cv2.imwrite("{}/{}-{}-test2_1.png".format(LOGS_PATH,
                                                          self.model_name, episode_counter),
                            output.cpu().detach().numpy()[0][0] * 255)

            self.logger.warning("Loss for episode {} is {}".format(episode_counter, loss))
            prev_loss = loss.item()
            episode_counter += 1

        np.save("{}-{}-{}".format(LOSSES_FILE_PATH, self.model_name,
                                  str(episode_counter)), np.array(self.losses))

        torch.save(
            {
                'optimizer': self.optimizer.state_dict(),
                'model': self.model.state_dict()
            }, "{}-{}-state-{}".format(MODEL_FILE_PATH, self.model_name, episode_counter))


def train_single_model(training_data_loader, model_name, auto_encoder_model=None,
                       optimizer=None, losses=None, start_episode_counter=0):

    if auto_encoder_model is None:
        auto_encoder_model = AutoEncoderModel()
        auto_encoder_model = auto_encoder_model.float()
        auto_encoder_model = auto_encoder_model.to(DEVICE)
    else:
        auto_encoder_model.train()
    auto_encoder = AutoEncoder(model_name, auto_encoder_model, device=DEVICE)
    auto_encoder.post_setup(optimizer)
    if losses is not None:
        auto_encoder.losses = losses.tolist()

    # all_training_data_sampler = TrainingDataSampler(training_data_loader, EPISODES_COUNT)
    auto_encoder.train_batches(EPISODES_COUNT, training_data_loader,
                               start_episodes_counter=start_episode_counter)


def get_model(training_data_loader, episodes_count, device, model_name, exclusion_filter):
    model = AutoEncoderModel()
    model = model.float()
    model = model.to(device)
    data_sampler = TrainingDataSampler(training_data_loader, episodes_count, exclusion_filter)
    auto_encoder = AutoEncoder(model_name, model, device=device)
    auto_encoder.post_setup()
    return threading.Thread(target=auto_encoder.train_batches, args=(episodes_count, data_sampler))


def train_multiple_models(training_data_loader):

    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print("Using multiple GPUs!")
        devices = [torch.device("cuda:0"), torch.device("cuda:1"),
                   torch.device("cuda:2"), torch.device("cuda:3")]

    elif not torch.cuda.is_available():
        print("Warning! Cannot execute multiple models on CPU")
        devices = [torch.device("cpu"), torch.device("cpu"),
                   torch.device("cpu"), torch.device("cpu")]
    else:
        raise Exception("Trying to run multiple models on single GPU")

    full_auto_encoder_thread = get_model(training_data_loader,
                                         EPISODES_COUNT, devices[0], 'all', None)

    exclude_avenue_thread = get_model(training_data_loader,
                                      EPISODES_COUNT, devices[1], 'exclude_avenue', {"avenue"})

    exclude_enter_thread = get_model(training_data_loader,
                                     EPISODES_COUNT, devices[2], 'exclude_enter', {"enter"})

    exclude_exit_thread = get_model(training_data_loader,
                                    EPISODES_COUNT, devices[3], 'exclude_exit', {"exit"})

    threads1 = [full_auto_encoder_thread, exclude_avenue_thread, exclude_enter_thread,
                exclude_exit_thread]
    print("Starting Thread 1")
    for thread in threads1:
        thread.start()

    for thread in threads1:
        thread.join()

    print("Done with Thread 1")

    print("Starting Thread 2")

    exclude_ped1_thread = get_model(training_data_loader,
                                    EPISODES_COUNT, devices[0], 'exclude_ped1', {"ped1"})
    exclude_ped2_thread = get_model(training_data_loader,
                                    EPISODES_COUNT, devices[1], 'exclude_ped2', {"ped2"})
    include_avenue_thread = get_model(training_data_loader,
                                      EPISODES_COUNT, devices[2], 'avenue',
                                      {"ped1", "ped2", "enter", "exit"})
    include_enter_thread = get_model(training_data_loader, EPISODES_COUNT, devices[3], 'enter',
                                     {"ped1", "ped2", "exit", "avenue"})

    threads2 = [exclude_ped1_thread, exclude_ped2_thread, include_avenue_thread,
                include_enter_thread]

    for thread in threads2:
        thread.start()

    for thread in threads2:
        thread.join()

    print("Done with Thread 2")

    print("Starting Thread 3")

    include_exit_thread = get_model(training_data_loader, EPISODES_COUNT, devices[1],
                                    'exit', {'ped2', 'enter', 'ped1', 'avenue'})

    include_ped1_thread = get_model(training_data_loader, EPISODES_COUNT, devices[2],
                                    'ped1', {'ped2','enter', 'exit', 'avenue'})

    include_ped2_thread = get_model(training_data_loader, EPISODES_COUNT, devices[3],
                                    'ped2', {'ped1', 'enter', 'exit', 'avenue'})

    threads3 = [include_exit_thread, include_ped1_thread, include_ped2_thread]

    for thread in threads3:
        thread.start()

    for thread in threads3:
        thread.join()

    print("Done with Thread 3")


def load_existing_model_and_test():
    auto_encoder_model, optimizer, _ = get_existing_model('all')
    auto_encoder = AutoEncoder(DATA_SET_TYPE, auto_encoder_model, optimizer)
    auto_encoder.model.eval()
    model_tester = ModelTester(auto_encoder.model, "avenue/testing_videos/15.avi")
    model_tester.test()


def load_existing_model_and_train(model_name, training_data_loader):
    auto_encoder_model, optimizer, losses = get_existing_model(model_name,
                                                               start_episode_counter=20000,
                                                               fetch_losses=True)
    train_single_model(training_data_loader, model_name,
                       auto_encoder_model=auto_encoder_model, optimizer=optimizer, losses=losses,
                       start_episode_counter=20000)


def get_existing_model(model_name, start_episode_counter=0, fetch_losses=False):
    model_state_dict = torch.load(PATH, map_location='cpu')
    auto_encoder_model = AutoEncoderModel()
    auto_encoder_model = auto_encoder_model.float()
    auto_encoder_model.load_state_dict(model_state_dict['model'])
    auto_encoder_model = auto_encoder_model.to(DEVICE)
    optimizer = Adagrad(auto_encoder_model.parameters(), lr=LEARNING_RATE, weight_decay=0.0005)
    optimizer.load_state_dict(model_state_dict['optimizer'])
    losses = None
    if fetch_losses:
        losses = np.load("{}-{}-{}.npy".format(LOSSES_FILE_PATH, model_name, str(start_episode_counter)))

    return auto_encoder_model, optimizer, losses


if __name__ == "__main__":

    if RUN_MODE == "save" or RUN_MODE == "load_and_save":
        files = []
        os.makedirs(LOGS_PATH)
        for file in listdir(INPUT_DATA_PATH):
            full_file_path = join(INPUT_DATA_PATH, file)
            if isfile(full_file_path):
                files.append(full_file_path)

        if LOAD_ALL_MEMORY:
            all_training_data_loader = AllTrainingDatasetLoader(PARENT_DIR_LIST)
            all_training_data_loader.load_all_training_data()
            train_multiple_models(all_training_data_loader)
        else:
            file_sampler = TrainingFileManager(PARENT_DIR_LIST, EPISODES_COUNT)
            if RUN_MODE == "load_and_save":
                load_existing_model_and_train('all', file_sampler)
            else:
                train_single_model(file_sampler, 'all')

    else:
        load_existing_model_and_test()
        print("hello")
