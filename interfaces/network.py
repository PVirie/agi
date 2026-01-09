import abc

class Policy_Network(abc.ABC):
    
    @abc.abstractmethod
    def get_log_probability(self, context, action, valid_actions=None, target_action=None, only_logprob_components=False):
        pass

    @abc.abstractmethod
    def get_action(self, context, action, valid_actions=None):
        pass

    @abc.abstractmethod
    def unpack_action(self, packed_action):
        pass

    @abc.abstractmethod
    def pack_action(self, b_int=None, b_ext=None, b_content=None):
        pass



class Value_Network(abc.ABC):

    @abc.abstractmethod
    def get_latest_value(self, context):
        pass

    @abc.abstractmethod
    def get_value(self, context):
        pass


class Q_Network(abc.ABC):
    
    @abc.abstractmethod
    def get_q1_values(self, context, action, valid_actions=None):
        pass

    @abc.abstractmethod
    def get_q2_values(self, context, action, valid_actions=None):
        pass

    @abc.abstractmethod
    def get_q1_target_values(self, context, action, valid_actions=None):
        pass

    @abc.abstractmethod
    def get_q2_target_values(self, context, action, valid_actions=None):
        pass
    
