import abc

class Agent(abc.ABC):

    @abc.abstractmethod
    def reset(self):
        pass

    @abc.abstractmethod
    def choose_action(self, 
                      last_idles, next_dones, last_truncates, last_resets, 
                      latest_frames, rewards, next_available_actions, 
                      force_train=False):
        pass