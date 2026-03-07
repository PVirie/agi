import abc

class Policy_Network(abc.ABC):
    
    @abc.abstractmethod
    def get_action(self, context, valid_actions=None):
        pass

    @abc.abstractmethod
    def get_log_probability(self, context, selected_action, valid_actions=None):
        pass

    @abc.abstractmethod
    def get_log_probability_with_aux_loss(self, context, selected_action, valid_actions=None):
        pass

    @abc.abstractmethod
    def unpack_action(self, packed_action):
        pass

    @abc.abstractmethod
    def pack_context(self, b_reward=None, b_position=None, b_content=None):
        pass


class Policy_Index_Network(Policy_Network):
    
    @abc.abstractmethod
    def get_action(self, context, indices, valid_actions=None):
        pass

    @abc.abstractmethod
    def get_log_probability(self, context, indices, selected_action, valid_actions=None):
        pass

    @abc.abstractmethod
    def get_log_probability_with_aux_loss(self, context, indices, selected_action, valid_actions=None):
        pass



class Value_Network(abc.ABC):

    @abc.abstractmethod
    def get_value(self, context):
        pass


class Value_Index_Network(Value_Network):

    @abc.abstractmethod
    def get_value(self, context, indices):
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
    
