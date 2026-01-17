import abc

class Policy_Network(abc.ABC):
    
    @abc.abstractmethod
    def get_log_probability(self, context, selected_action, valid_actions=None):
        pass

    @abc.abstractmethod
    def get_action(self, context, valid_actions=None):
        pass

    @abc.abstractmethod
    def unpack_action(self, packed_action):
        pass

    @abc.abstractmethod
    def pack_context(self, b_reward=None, b_position=None, b_content=None):
        pass


class Value_Network(abc.ABC):

    @abc.abstractmethod
    def get_value(self, context):
        pass


class Q_Network(abc.ABC):
    
    @abc.abstractmethod
    def get_q1_values(self, context, action):
        pass

    @abc.abstractmethod
    def get_q2_values(self, context, action):
        pass

    @abc.abstractmethod
    def get_q1_target_values(self, context, action):
        pass

    @abc.abstractmethod
    def get_q2_target_values(self, context, action):
        pass
    
