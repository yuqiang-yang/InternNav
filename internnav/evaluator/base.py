from internnav.agent.utils.client import AgentClient
from internnav.configs.evaluator import EvalCfg
from internnav.env import Env


class Evaluator:
    """
    Base class of all evaluators.
    """

    evaluators = {}

    def __init__(self, config: EvalCfg):
        self.config = config
        self.env = Env.init(config.env, config.task)
        self.agent = AgentClient(config.agent)

    def eval(self):
        raise NotImplementedError

    @classmethod
    def register(cls, evaluator_type: str):
        """
        Register a evaluator class.
        """

        def decorator(evaluator_class):
            cls.evaluators[evaluator_type] = evaluator_class

        return decorator

    @classmethod
    def init(cls, config: EvalCfg):
        """
        Init a evaluator instance from a config.
        """
        return cls.evaluators[config.env.env_type](config)
