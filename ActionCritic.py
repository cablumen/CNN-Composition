import numpy as np
import tensorflow as tf

import Settings
from Settings import LogLevel

GAMMA = 0.95

class ActionCritic:
    def __init__(self, logger, data_manager, architecture):
        self.__logger = logger
        self.__data_manager = data_manager
        self.__name = architecture.name

        # TODO: implement reset for CNN
        critic_model = tf.keras.models.clone_model(architecture.value)
        critic_model.compile(optimizer='adam', loss='mse', metrics=['mse'], jit_compile=True)
        self.__model = critic_model

        self.__model_verbosity = True if Settings.LogLevel == 0 else False

    def get_name(self):
        return self.__name

    @tf.function
    def train(self):
        self.__logger.print("ActionCritic(train): " + self.__name, LogLevel.INFO)
        if self.__data_manager.is_critic_training_ready():
            states_frames, next_states_frames, rewards, dones = self.__data_manager.get_critic_data()
            targets = rewards + GAMMA * self.predict_batch(next_states_frames) * (1 - dones)
            self.__model.fit(states_frames, targets, epochs=Settings.EPOCHS, verbose=self.__model_verbosity)
        
    def predict(self, state_frame):
        self.__logger.print("ActionCritic(predict): " + self.__name, LogLevel.INFO)
        return self.__model.predict(state_frame, verbose=self.__model_verbosity)

    def predict_batch(self, state_batch):
        self.__logger.print("ActionCritic(predict_batch): " + self.__name, LogLevel.INFO)
        return np.array([prediction[0] for prediction in self.__model.predict(state_batch, batch_size=Settings.BATCH_SIZE, verbose=self.__model_verbosity)])
