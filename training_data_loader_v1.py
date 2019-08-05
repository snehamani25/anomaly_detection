from queue import Queue
import functools
from threading import Thread
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import os
from multiprocessing.pool import ThreadPool
import datetime

BATCH_SIZE = 32


class RegularizationDataset(Dataset):
    TRAINING_VIDEOS = "training_videos"
    INPUT_DATA = "input_data"
    MAX_CACHE_SIZE = 10

    def __init__(self, data_folders):

        self.file_count_list = []
        self.total_count = 0
        for folder in data_folders:
            training_videos_path = "{}/{}".format(folder, RegularizationDataset.INPUT_DATA)
            for file in os.listdir(training_videos_path):
                full_file_path = os.path.join(training_videos_path, file)
                if os.path.isfile(full_file_path):
                    tokens = file.split("_")
                    count = int(tokens[len(tokens)-1].split(".")[0])
                    self.file_count_list.append((count, full_file_path))
                    self.total_count += count

        self.file_count_list.sort()
        self.file_map = {(item[1], item[0]) for item in self.file_count_list}

    def __len__(self):
        return self.total_count

    @functools.lru_cache(maxsize=MAX_CACHE_SIZE)
    def get_batches_from_file(self, file_name):
        data = np.load(file_name)
        print("Loaded file ", file_name, datetime.datetime.now())
        return data

    def __getitem__(self, index):
        print("Fetching index", index)
        curr_index = 0
        batch_index = 0
        prev_curr_index = None

        while curr_index <= index:
            prev_curr_index = curr_index
            curr_index += self.file_count_list[batch_index][0]
            batch_index += 1

        if prev_curr_index is None:
            batches = self.get_batches_from_file(self.file_count_list[0][1])
            return batches[index]

        batch_index -= 1
        batches = self.get_batches_from_file(self.file_count_list[batch_index][1])
        return batches[index-prev_curr_index]


class TrainingFileManager(Thread):

    MAX_QUEUE_SIZE = 10
    NUM_WORKERS = 1

    def __init__(self, parent_folders, episodes_count):
        super(TrainingFileManager, self).__init__()
        self.queue = Queue(maxsize=self.MAX_QUEUE_SIZE)
        self.done = False
        self.dataset_loader = DataLoader(RegularizationDataset(parent_folders),
                                         batch_size=BATCH_SIZE, shuffle=True,
                                         num_workers=self.NUM_WORKERS)
        self.curr_episode_count = 0
        self.episodes_count = episodes_count

    def run(self):
        while self.curr_episode_count < self.episodes_count:

            cuboids = None
            # print("Fetching a batch of data", datetime.datetime.now())
            for batch in self.dataset_loader:
                print("Fetched a batch of data", datetime.datetime.now())
                cuboids = torch.from_numpy(np.array(batch)).double()
                cuboids = cuboids.float()

                # If the queue is not full, add the cuboids sampled
                if not self.queue.full():
                    # print("Fetching a batch of data", datetime.datetime.now())
                    self.queue.put(cuboids)
                    self.curr_episode_count += 1
                # Else, the queue is full, break from the current iterator
                else:
                    break

            # Wait for the queue to get an empty slot and add the outstanding cuboid
            self.queue.put(cuboids)
            self.curr_episode_count += 1

    def get_training_data(self):
        return self.queue.get()

    def stop(self):
        self.done = True


class TrainingDataSampler(Thread):
    MAX_QUEUE_SIZE = 10
    NUM_PROCESSORS = 100  # Number of files to load in parallel

    def __init__(self, episodes_count):
        super(TrainingDataSampler, self).__init__()
        self.queue = Queue(maxsize=self.MAX_QUEUE_SIZE)
        self.done = False
        self.inputs = None
        self.curr_episode_count = 0
        self.total_count = 0
        self.episodes_count = episodes_count

    @staticmethod
    def _read_single_batch_file(file_name):
        print("Loading ", file_name)
        return np.load(file_name)

    def load_all_training_data(self, files_list):
        print("{} : Loading {} files".format(datetime.datetime.now(), len(files_list)))
        files_read = 0
        files_rem = len(files_list)
        inputs = []
        pool = ThreadPool(processes=TrainingDataSampler.NUM_PROCESSORS)
        while files_read < len(files_list):
            threads = []

            thread_count = 0
            while thread_count < min(TrainingDataSampler.NUM_PROCESSORS, files_rem):

                threads.append(pool.apply_async(TrainingDataSampler._read_single_batch_file,
                                                (files_list[files_read + thread_count],)))
                thread_count += 1

            for curr_thread_count in range(0, min(TrainingDataSampler.NUM_PROCESSORS, files_rem)):
                batch_data = threads[curr_thread_count].get()
                self.total_count += batch_data.shape[0]
                inputs.append(batch_data)

            files_read += (thread_count + 1)
            files_rem -= (thread_count + 1)
        self.inputs = inputs
        # inputs = np.concatenate(inputs, axis=0)
        print("{} : Loaded {} files".format(datetime.datetime.now(), len(files_list)))

    def get_batch_index(self, index):
        curr_index = 0
        batch_index = 0
        prev_curr_index = None

        while curr_index <= index:
            prev_curr_index = curr_index
            curr_index += self.inputs[batch_index].shape[0]
            batch_index += 1

        if prev_curr_index is None:
            batches = self.inputs[batch_index].shape[0]
            return batches[index]

        batch_index -= 1
        return  self.inputs[batch_index][index-prev_curr_index]

    def run(self):
        while self.curr_episode_count < self.episodes_count:
            indexes = np.random.choice(self.total_count, BATCH_SIZE)
            cuboids_batches = np.array([self.get_batch_index(index) for index in indexes])
            cuboids = torch.from_numpy(cuboids_batches).double()
            cuboids = cuboids.float()
            self.queue.put(cuboids)
            self.curr_episode_count += 1

    def get_training_data(self):
        return self.queue.get()

    def stop(self):
        self.done = True

