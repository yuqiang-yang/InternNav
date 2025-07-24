import numpy as np
from internnav.projects.internutopia_vln_extension.configs.metrics.vln_pe_metrics import VLNPEMetricCfg
from internnav.projects.internutopia_vln_extension.configs.tasks.vln_eval_task import VLNEvalTaskCfg
from internutopia.core.task.metric import BaseMetric

from internnav.projects.internutopia_vln_extension.configs.metrics.vln_pe_metrics import (
    VLNPEMetricCfg,
)

XY_DISTANCE_CLOSE_THRESHHOLD = 1.0


@BaseMetric.register('VLNPEMetric')
class VLNPEMetrics(BaseMetric):
    """
    Calculate the success of this episode
    """

    def __init__(self, config: VLNPEMetricCfg, task_config: VLNEvalTaskCfg):  # runtime 取数据
        self.config = config
        super().__init__(config, task_config)
        self.shortest_path_length_calc = self.config.shortest_to_goal_distance
        self.success_distance = self.config.success_distance
        self.path_data = task_config.data
        self.shortest_path_length = self.path_data['info']['geodesic_distance']
        self.goal_position = self.path_data['reference_path'][-1]
        self.current_path_length = 0
        self.pred_traj_list = [[]]
        self.metrics = {}
        self.sim_step = 0
        self.fail_reason = []
        self.ne = None
        self.prev_position = None
        self.reset()

    def reset(self):
        pass

    def _calc_ndtw(self):
        dtw_threshold = self.success_distance
        # 计算路径间的累积DTW距离
        dtw_distance = 0.0
        trajectory = []
        if len(self.pred_traj_list[0]) > 0:
            trajectory = np.array(self.pred_traj_list[0])[:, :2]  # 只取x,y坐标
            reference_path = np.array(self.path_data['reference_path'])[:, :2]

            for point in trajectory:
                # 找到参考路径上最近的点
                min_dist = float('inf')
                for ref_point in reference_path:
                    dist = np.linalg.norm(point - ref_point)
                    min_dist = min(min_dist, dist)
                # 累加DTW距离,使用高斯函数进行归一化
                dtw_distance += np.exp(-(min_dist**2) / (2 * dtw_threshold**2))

        # 归一化DTW得分
        ndtw = dtw_distance / len(trajectory) if len(trajectory) > 0 else 0.0
        return ndtw

    def update(self, task_obs: dict):  # step更新相关
        robot_name = list(task_obs.keys())[0]
        obs = task_obs[robot_name]
        current_position = obs['globalgps']
        self.fail_reason = obs['fail_reason'] if 'fail_reason' in obs else ''

        # 更新步骤计数
        self.sim_step += 1  
        # 计算当前path_length
        if self.prev_position is not None:  # 初始化，warm_up会变位置
            self.current_path_length += np.linalg.norm(current_position[:2] - self.prev_position[:2])  # 总路程长度 只要xy
        else:
            self.pred_traj_list[0].append(current_position)  
        self.prev_position = current_position
        # 当前的
        if obs['finish_action']:
            # 添加轨迹数组
            self.pred_traj_list[0].append(current_position)  # 轨迹数组 考虑计算量完成轨迹再更新

            # 计算NE, 每回合都需要
            self.ne = np.linalg.norm(current_position[:2] - self.goal_position[:2])

            # OSR 看曾经中过没有
            self.shortest_path_length_calc = min(self.shortest_path_length_calc, self.ne)

    # def calc(self, task_info: dict): #最后
    def calc(self):  # 最后
        self.metrics['shortest_path_length'] = self.shortest_path_length
        # success distance 计算
        self.metrics['NE'] = self.ne
        self.metrics['success'] = float(self.ne < self.success_distance)

        # OSR 看曾经中过没有
        self.metrics['osr'] = float(self.shortest_path_length_calc < self.success_distance)

        # TL计算,轨迹总长度
        self.metrics['TL'] = self.current_path_length

        # SPL
        self.metrics['spl'] = (
            self.metrics['success']
            * self.shortest_path_length
            / max(self.current_path_length, self.shortest_path_length)
            if self.current_path_length > 0
            else 0
        )

        # NDTW计算
        self.metrics['ndtw'] = self._calc_ndtw()
        self.metrics['steps'] = self.sim_step

        self.metrics['episode_id'] = self.path_data['episode_id']  # episode ID
        self.metrics['trajectory_id'] = self.path_data['trajectory_id']  # 轨迹 ID
        self.metrics['fail_reason'] = self.fail_reason
        self.metrics['reference_path'] = self.path_data['reference_path']
        self.metrics['reference_path'] = np.array(self.metrics['reference_path']).tolist()

        return [self.metrics]
