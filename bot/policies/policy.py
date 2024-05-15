from abc import ABC, abstractmethod

class Policy(ABC):
    def __init__(self, **kwargs):
        """
        Constructor for the Policy class. It can take flexible parameters.
        Contestants are free to maintain internal state and use any auxiliary data or information
        within this class.
        """
        super().__init__()

    def act(self, external_state, internal_state):
        """
        Method to be called when the policy needs to make a decision.

        :param external_state: A dictionary containing the current market data.
        :param internal_state: A dictionary containing the internal state of the policy.
        :return: A tuple (amount to route from solar panel to battery: int, amount to charge battery from grid: int). Note: any solar power not routed to the battery goes directly to the grid and you earn the current spot price.
        """
        pass
    
    @abstractmethod
    def load_historical(self, external_states):
        """
        Load historical data to the policy. This method is called once before the simulation starts.

        :param external_states: A list of dictionaries containing historical market data to be used as relevant context
        when acting later.
        """
        pass
