import numpy as np
import random

import Settings


class DataManager:
    def __init__(self):
        # {action_index -> ([[state_frame, predicted_next_state_frame, next_state_frame, reward, done], ...])
        self.__action_data = {}
        self.reset()
    
    def reset(self):
        for action_index in range(Settings.ACTION_SIZE):
            self.__action_data[action_index] = []

    def put_action_data(self, action_index, action_record):
        return self.__action_data[action_index].append(action_record)

    #       actor data
    def is_actor_training_ready(self, action_index):
        return self.__get_action_datasize(action_index) >= Settings.BATCH_SIZE * 2
        
    def get_actor_data(self, action_index):
        train_x = np.empty((Settings.BATCH_SIZE, Settings.RENDER_PIXEL_HEIGHT, Settings.RENDER_PIXEL_WIDTH, Settings.RENDER_PIXEL_DEPTH), dtype=np.float32)
        train_y = np.empty((Settings.BATCH_SIZE, Settings.RENDER_PIXEL_HEIGHT, Settings.RENDER_PIXEL_WIDTH, Settings.RENDER_PIXEL_DEPTH), dtype=np.float32)
        test_x = np.empty((Settings.BATCH_SIZE, Settings.RENDER_PIXEL_HEIGHT, Settings.RENDER_PIXEL_WIDTH, Settings.RENDER_PIXEL_DEPTH), dtype=np.float32)
        test_y = np.empty((Settings.BATCH_SIZE, Settings.RENDER_PIXEL_HEIGHT, Settings.RENDER_PIXEL_WIDTH, Settings.RENDER_PIXEL_DEPTH), dtype=np.float32)

        # randomly generate action indicies to sample
        random_indicies = random.sample(range(0, self.__get_action_datasize(action_index)), Settings.BATCH_SIZE * 2)
        sampled_training_indicies = random_indicies[0:Settings.BATCH_SIZE]
        sampled_validation_indicies = random_indicies[Settings.BATCH_SIZE:]

        # populate test sample
        batch_index = 0
        for random_index in sampled_training_indicies:
            action_data = self.__action_data[action_index][random_index]
            state_frame = action_data[0][0]
            next_state_frame = action_data[2][0]
            frame_delta = np.subtract(next_state_frame, state_frame)
            train_x[batch_index] = state_frame
            train_y[batch_index] = frame_delta
            batch_index += 1

        
        # populate validation sample
        batch_index = 0
        for random_index in sampled_validation_indicies:
            action_data = self.__action_data[action_index][random_index]
            state_frame = action_data[0][0]
            next_state_frame = action_data[2][0]
            frame_delta = np.subtract(next_state_frame, state_frame)
            test_x[batch_index] = state_frame
            test_y[batch_index] = frame_delta
            batch_index += 1

        return train_x, train_y, test_x, test_y

    #       critic data
    def is_critic_training_ready(self):
        is_ready = True
        action_batchsize = int(Settings.BATCH_SIZE / Settings.ACTION_SIZE)
        for action_index in range(Settings.ACTION_SIZE):
            is_ready = is_ready and (self.__get_action_datasize(action_index) >= action_batchsize)

        return is_ready

    def get_critic_data(self):
        states_frames = np.empty((Settings.BATCH_SIZE, Settings.RENDER_PIXEL_HEIGHT, Settings.RENDER_PIXEL_WIDTH, Settings.RENDER_PIXEL_DEPTH), dtype=np.float32)
        next_states_frames = np.empty((Settings.BATCH_SIZE, Settings.RENDER_PIXEL_HEIGHT, Settings.RENDER_PIXEL_WIDTH, Settings.RENDER_PIXEL_DEPTH), dtype=np.float32)
        rewards = np.empty(Settings.BATCH_SIZE, dtype=float)
        dones = np.empty(Settings.BATCH_SIZE, dtype=np.int32)
    
        batch_index = 0
        action_sample_size = int(Settings.BATCH_SIZE / Settings.ACTION_SIZE)
        for action_index in range(Settings.ACTION_SIZE):
            # randomly generate action indicies to sample
            random_indicies = random.sample(range(0, self.__get_action_datasize(action_index)), action_sample_size)

            for random_index in random_indicies:
                action_data = self.__action_data[action_index][random_index]
                states_frames[batch_index] = action_data[0][0]
                next_states_frames[batch_index] = action_data[2][0]
                rewards[batch_index] = action_data[3]
                dones[batch_index] = action_data[4]
                batch_index += 1

        return states_frames, next_states_frames, rewards, dones

    #       private functions
    def __get_action_datasize(self, action_index):
        return len(self.__action_data[action_index])