from typing import Any, Dict, List

from internutopia.core.config import Config, SimConfig
from internutopia.core.config.distribution import RayDistributionCfg
from internutopia.core.vec_env import Env

from internnav.configs.evaluator import EnvCfg, TaskCfg
from internnav.env import base
from internnav.projects.internutopia_vln_extension.configs.tasks.vln_eval_task import (
    VLNEvalTaskCfg,
)
from internnav.projects.internutopia_vln_extension import import_extensions


@base.Env.register('vln_pe')
class VlnPeEnv(base.Env):
    def __init__(self, env_config: EnvCfg, task_config: TaskCfg):
        super().__init__(env_config, task_config)
        env_settings = self.env_config.env_settings
        task_settings = self.task_config.task_settings
        config = Config(
            simulator=SimConfig(**env_settings),
            env_num=task_settings['env_num'],
            env_offset_size=task_settings['offset_size'],
            task_configs=task_settings['episodes'],
        )
        if 'distribution_config' in env_settings:
            distribution_config=RayDistributionCfg(**env_settings['distribution_config'])
            config = config.distribute(distribution_config)
        import_extensions()

        self.env = Env(config)

    def reset(self, reset_index=None):
        return self.env.reset(reset_index)

    def step(self, action: List[Any]):
        return self.env.step(action)

    def is_running(self):
        return True

    def close(self):
        print('Vln Env close')
        self.env.close()

    def render(self):
        self.env.render()

    def get_observation(self) -> Dict[str, Any]:
        return self.env.get_observations()

    def get_info(self) -> Dict[str, Any]:
        pass
