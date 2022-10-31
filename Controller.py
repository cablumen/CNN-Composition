import cv2
import gym
import numpy as np

from Architectures import Architectures
from ActionDictionary import ActionDictionary
from MetricCalculator import MetricCalculator
import Settings

class Controller:
    def __init__(self):
        # self.__env = gym.make("CartPole-v1", render_mode='human')
        self.__env = gym.make("CartPole-v1")
        # self.__metric_calculator  = MetricCalculator(self.__env)

        self.__experiments = [(Architectures.CONV_80x80, Architectures.REWARD_80x80)] # format: (sub_model_architecture, reward_architecture)
        self.__action_dicts = []
        for sub_model_architecture, reward_architecture in self.__experiments:
            self.__action_dicts.append(ActionDictionary(sub_model_architecture, reward_architecture))

        for action_dict in self.__action_dicts:
            self.epsilon_exploration(action_dict, 200)

    # execute best predicted action with increasing probability
    def epsilon_exploration(self, action_dictionary, episode_count):
        print("\nController(epsilon_exploration): submodel:" + action_dictionary.get_submodel_name() + " reward:" + action_dictionary.get_reward_model_name())
        epsilon = Settings.EPSILON_START

        rewards_list = []
        for episode in range(episode_count):
            self.__env.reset()
            state_frame = self.get_env_frame()

            reward_for_episode = 0
            for step in range(Settings.MAX_TRAINING_STEPS):
                if np.random.rand() < epsilon:
                    action_index, predicted_next_state_frame = action_dictionary.predict_random_action(state_frame) 
                else:
                    action_index, predicted_next_state_frame = action_dictionary.predict_actions(state_frame)

                next_state, reward, done, info = self.__env.step(action_index)
                next_state_frame = self.get_env_frame()
                action_dictionary.put_action_data(action_index, [state_frame, predicted_next_state_frame, next_state_frame, reward, done])
                reward_for_episode += reward
                state_frame = next_state_frame
                
                action_dictionary.train_models()

                if done:
                    break

            # reduce probability of random action
            if epsilon > Settings.EPSILON_MIN:
                epsilon *= Settings.EPSILON_DECAY

            rewards_list.append(reward_for_episode)
            last_rewards_mean = np.mean(rewards_list[-30:])
            print("\tEpisode: ", episode, " || Reward: ", reward_for_episode, " || Average Reward: ", last_rewards_mean)
        
        return rewards_list

    def get_env_frame(self):
        frame_rgb = self.__env.render(mode = 'rgb_array')
        cv2.imshow("cartpole", frame_rgb)
        
        img_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        img_rgb_resized = cv2.resize(img_rgb, (Settings.RENDER_PIXEL_WIDTH, Settings.RENDER_PIXEL_HEIGHT), interpolation=cv2.INTER_CUBIC)
        img_rgb_resized = img_rgb_resized / 255

        # added two axes, one for pixel depth=1, and one for batched array
        img_with_axes = np.expand_dims(np.expand_dims(img_rgb_resized, axis=2), axis=0)
        return img_with_axes


if __name__ == '__main__':
    Controller()