import numpy as np
from collections import deque

class LossesData:
    FILE_NAME = "all_T20/model/losses-all-40000.npy"
    MAX_SIZE = 5

    def __init__(self):
        self.losses = np.load(self.FILE_NAME)
        self.queue = deque(maxlen=self.MAX_SIZE)

    def check_if_increasing_window(self):
        for index in range(0, self.losses.shape[0]):
            #print(index, self.losses[index])
            self.queue.append(self.losses[index])
            if len(self.queue) < self.MAX_SIZE:
                continue

            curr_index = 0
            while curr_index < len(self.queue)-1 and self.queue[curr_index+1] >= self.queue[curr_index]:
                curr_index += 1

            if curr_index == len(self.queue)-1:
                print("All are increasing")


if __name__ == "__main__":
    losses_data = LossesData()
    losses_data.check_if_increasing_window()
