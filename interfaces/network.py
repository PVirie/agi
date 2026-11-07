import abc


class Actor_Network(abc.ABC):
    
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


class Policy_Network(abc.ABC):
    
    @abc.abstractmethod
    def get_log_probability(self, context, selected_action, valid_actions=None):
        pass

    @abc.abstractmethod
    def get_log_probability_with_aux_loss(self, context, selected_action, valid_actions=None):
        pass


class Policy_Value_Network(abc.ABC):
    
    @abc.abstractmethod
    def get_log_probability_with_value(self, context, selected_action, valid_actions=None):
        pass

    @abc.abstractmethod
    def get_log_prob_with_value_aux(self, context, selected_action, valid_actions=None):
        """
        get_log_probability_with_value plus the auxiliary loss, computed in one forward pass.
        context: (batch, context_size + 1, ...); selected_action: (batch, context_size, ...)
        Returns (log_probs, entropy, values, aux_loss): values cover context_size + 1 steps;
        aux_loss is the unreduced auxiliary loss of shape (batch, context_size), or None if the policy has none.
        """
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
    
