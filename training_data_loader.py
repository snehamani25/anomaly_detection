from queue import Queue
from threading import Thread
import numpy as np
import torch
import os
from multiprocessing.pool import ThreadPool
import datetime

BATCH_SIZE = 32


class TrainingFileManager(Thread):

    MAX_QUEUE_SIZE = 10
    NUM_WORKERS = 1
    TRAINING_VIDEOS = "training_videos"
    INPUT_DATA = "input_data"
    MAX_CACHE_SIZE = 10
    NUM_PROCESSORS = 320

    def __init__(self, parent_folders, episodes_count, exclusion_filter=None):
        super(TrainingFileManager, self).__init__()
        self.queue = Queue(maxsize=self.MAX_QUEUE_SIZE)
        self.done = False
        self.curr_episode_count = 0
        self.episodes_count = episodes_count

        if exclusion_filter is None:
            exclusion_filter = set()

        self.file_count_list = []
        self.total_count = 0
        for folder in parent_folders:
            training_videos_path = "{}/{}".format(folder, self.INPUT_DATA)
            for file in os.listdir(training_videos_path):
                full_file_path = os.path.join(training_videos_path, file)
                if os.path.isfile(full_file_path):
                    tokens = file.split("_")
                    if tokens[0] in exclusion_filter:
                        continue
                    count = int(tokens[len(tokens) - 1].split(".")[0])
                    self.file_count_list.append(full_file_path)
                    self.total_count += count

        self.file_count_list.sort()
        self.pool = ThreadPool(processes=self.NUM_PROCESSORS)
        self.cache = {}

    def _read_single_batch_file(self, file_name):
        if file_name in self.cache:
            return self.cache[file_name]

        # print("Loading ", file_name)
        data = np.load(file_name)
        if len(self.cache) >= self.MAX_CACHE_SIZE:
            del self.cache[next(iter(self.cache))]
        self.cache[file_name] = data
        return data

    def get_sample_indexes(self, batch_size):

        batches = np.zeros((batch_size,20,227,227,),dtype=np.float16)
        to_be_loaded_files = np.random.choice(self.file_count_list, batch_size)
        files_rem = batch_size
        thread_count = 0
        threads = []
        # print("{} fetching batches of size {}".format(datetime.datetime.now(), BATCH_SIZE))
        while thread_count < min(self.NUM_PROCESSORS, batch_size):
            threads.append(self.pool.apply_async(self._read_single_batch_file,
                                                 (to_be_loaded_files[thread_count],)))
            thread_count += 1

        for curr_thread_count in range(0, min(self.NUM_PROCESSORS, files_rem)):

            batch_data = threads[curr_thread_count].get()
            batches[curr_thread_count] = batch_data
        # print("Loaded a batch ", datetime.datetime.now())
        return batches

    def run(self):
        while self.curr_episode_count < self.episodes_count:
            thread_count = 0
            threads = []
            while thread_count < min(self.NUM_PROCESSORS, self.MAX_QUEUE_SIZE):
                threads.append(self.pool.apply_async(self.get_sample_indexes,
                                                     (self.MAX_QUEUE_SIZE,)))
                thread_count += 1

            for curr_thread_count in range(0, min(self.NUM_PROCESSORS, self.MAX_QUEUE_SIZE)):
                batch = threads[curr_thread_count].get()
                cuboids = torch.from_numpy(np.array(batch)).double()
                cuboids = cuboids.float()
                self.queue.put(cuboids)
                self.curr_episode_count += 1

    def get_training_data(self):
        return self.queue.get()

    def stop(self):
        self.done = True


class AllTrainingDatasetLoader:
    INPUT_DATA = "input_data"
    NUM_PROCESSORS = 100 # Number of files to load in parallel

    def __init__(self, data_folders):
        self.file_count_list = []
        self.total_count = 0
        self.data_set_type_count = {}
        self.batches_map = {}

        for folder in data_folders:
            training_videos_path = "{}/{}".format(folder, self.INPUT_DATA)
            for file in os.listdir(training_videos_path):
                full_file_path = os.path.join(training_videos_path, file)
                if os.path.isfile(full_file_path):
                    tokens = file.split("_")
                    count = int(tokens[len(tokens) - 1].split(".")[0])

                    if tokens[0] not in self.data_set_type_count:
                        self.data_set_type_count[tokens[0]] = count
                    else:
                        self.data_set_type_count[tokens[0]] += count

                    self.file_count_list.append((count, full_file_path))
                    self.total_count += count

        self.file_count_list.sort()

    def get_batch_details(self):
        return self.batches_map, self.data_set_type_count

    @staticmethod
    def _read_single_batch_file(file_name):
        print("Loading ", file_name)
        return np.load(file_name)

    def load_all_training_data(self):
        print("{} : Loading {} files".format(datetime.datetime.now(), len(self.file_count_list)))
        files_read = 0
        files_rem = len(self.file_count_list)
        pool = ThreadPool(processes=self.NUM_PROCESSORS)
        while files_read < len(self.file_count_list):
            threads = []

            thread_count = 0
            while thread_count < min(self.NUM_PROCESSORS, files_rem):

                threads.append(pool.apply_async(AllTrainingDatasetLoader._read_single_batch_file,
                                                (self.file_count_list[files_read + thread_count][1],)))
                thread_count += 1

            for curr_thread_count in range(0, min(self.NUM_PROCESSORS, files_rem)):

                batch_data = threads[curr_thread_count].get()
                file_name = self.file_count_list[curr_thread_count][1].split("/").pop()
                file_type = file_name.split("_")[0]

                if file_type in self.batches_map:
                    self.batches_map[file_type].append(batch_data)
                else:
                    self.batches_map[file_type] = [batch_data]

                self.total_count += batch_data.shape[0]

            files_read += (thread_count + 1)
            files_rem -= (thread_count + 1)
        print("{} : Loaded {} files".format(datetime.datetime.now(), len(self.file_count_list)))


class TrainingDataSampler(Thread):
    MAX_QUEUE_SIZE = 10

    def __init__(self, data_loader, episodes_count, exclusion_filter=None):
        super(TrainingDataSampler, self).__init__()
        self.queue = Queue(maxsize=self.MAX_QUEUE_SIZE)
        self.done = False
        self.curr_episode_count = 0
        self.total_count = 0
        self.episodes_count = episodes_count
        self.batches_map, self.batches_count = data_loader.get_batch_details()

        if exclusion_filter is None:
            exclusion_filter = set()

        self.available_keys = list(set(self.batches_map.keys()).difference(exclusion_filter))
        self.prob_map = np.array([self.batches_count[key] for key in self.available_keys])
        self.prob_map = self.prob_map / np.sum(self.prob_map)

    def get_sample_indexes(self, batch_size):

        batches = []
        data_set_types = np.random.choice(self.available_keys, batch_size, p=self.prob_map)
        for data_set_type in data_set_types:
            index = np.random.randint(0, self.batches_count[data_set_type])
            batches.append(self.get_batch_index(data_set_type, index))
        return np.array(batches)

    def get_batch_index(self, data_set_type, index):
        curr_index = 0
        batch_index = 0
        prev_curr_index = None
        inputs = self.batches_map[data_set_type]

        while curr_index <= index:
            prev_curr_index = curr_index
            curr_index += inputs[batch_index].shape[0]
            batch_index += 1

        if prev_curr_index is None:
            batches = inputs[batch_index].shape[0]
            return batches[index]

        batch_index -= 1
        return inputs[batch_index][index-prev_curr_index]

    def run(self):
        while self.curr_episode_count < self.episodes_count:
            cuboids_batches = self.get_sample_indexes(BATCH_SIZE)
            cuboids = torch.from_numpy(cuboids_batches).double()
            cuboids = cuboids.float()
            self.queue.put(cuboids)
            self.curr_episode_count += 1

    def get_training_data(self):
        return self.queue.get()

    def stop(self):
        self.done = True

