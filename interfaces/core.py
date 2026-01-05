import abc

class Core(abc.ABC):

    @abc.abstractmethod
    def get_latest_value(self, context, action):
        pass

    @abc.abstractmethod
    def get_action_and_value(self, context, action, valid_actions=None, use_action=False, use_grad=True):
        pass

    @abc.abstractmethod
    def get_log_probability(self, context, action, valid_actions=None, target_action=None, f_mask=None):
        pass

    @abc.abstractmethod
    def unpack_action(self, packed_action):
        pass

    @abc.abstractmethod
    def pack_action(self, b_int=None, b_ext=None, b_content=None):
        pass
