import abc

class Core(abc.ABC):
    
    @abc.abstractmethod
    def get_log_probability(self, context, action, valid_actions=None, target_action=None, f_mask=None):
        pass

    @abc.abstractmethod
    def unpack_action(self, packed_action):
        pass

    @abc.abstractmethod
    def pack_action(self, b_int=None, b_ext=None, b_content=None):
        pass

    @abc.abstractmethod
    def get_action(self, context, action, valid_actions=None):
        pass


class On_Policy_Core(Core):

    @abc.abstractmethod
    def get_latest_value(self, context, action):
        pass

    @abc.abstractmethod
    def get_value(self, context, action, valid_actions=None):
        pass


class Off_Policy_Core(Core):
    
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
    
