# 数据说明
## 训练数据

### metadata
与训练数据相关的id，参考路径，指令等信息的[metadata](../data/datasets/R2R_VLNCE_v1-3_one_scene/gather_data/train_gather_data.json)位于`data/datasets/R2R_VLNCE_v1-3_one_scene/gather_data/train_gather_data.json`,数据结构如下：
```json
{
  "scene_id": [
    {
      "episode_id": "episode identifier - int",
      "trajectory_id": "trajectory identifier - int", 
      "scene_id": "scene file path - string",
      "start_position": "initial robot position [x, y, z] - list[float]",
      "start_rotation": "initial robot orientation as quaternion [w, x, y, z] - list[float]",
      "info": {
        "geodesic_distance": "shortest path distance to goal - float"
      },
      "goals": [
        {
          "position": "target position [x, y, z] - list[float]",
          "radius": "acceptance radius around goal - float"
        }
      ],
      "instruction": {
        "instruction_text": "natural language navigation command - string",
        "instruction_tokens": "tokenized instruction padded to length 200 - list[int]"
      },
      "reference_path": "ground truth navigation path as list of positions - list[list[float]]",
      "original_start_position": "original start position before coordinate transformation - list[float]",
      "original_start_rotation": "original start rotation before coordinate transformation - list[float]",
      "scan": "scene identifier - string"
    },
    ...
  ],
  "scene_id": [
    ...
  ]
}
```

### 实际数据
根据metadata参考路径采集而来的实际数据位于[训练数据](https://xxxx)，以lmdb数据格式存储，数据结构如下
```json
{
    "camera_info": {
        "pano_camera_0": {
            "rgb": "<class 'numpy.ndarray'>: (bs, 256, 256, 3)",
            "depth": "<class 'numpy.ndarray'>: (bs, 256, 256)",
            "position": "<class 'numpy.ndarray'>: (bs, 3)",
            "orientation": "<class 'numpy.ndarray'>: (bs, 4)",
            "yaw": "<class 'numpy.ndarray'>: (bs,)"
        }
    },
    "robot_info": {
        "position": "<class 'numpy.ndarray'>: (bs, 3)",
        "orientation": "<class 'numpy.ndarray'>: (bs, 4)",
        "yaw": "<class 'numpy.ndarray'>: (bs,)"
    },
    "progress": "<class 'numpy.ndarray'>: (bs,)",
    "step": "<class 'numpy.ndarray'>: (bs,)",
    "action": "<class 'list'>"
}
```

### cma_dataset 转换后数据
为了更好的处理数据，与评测数据统一，在训练cma模型时，我们使用了[cma_dataset](../internnav/dataset/cma_dataset.py)，将原始数据转换成如下结构
```json
{
  "instruction": "tokenized natural language navigation command - numpy.ndarray(bs, 200)",
  "progress": "navigation task completion progress [0-1] - numpy.ndarray(bs,)",
  "globalgps": "current robot position in world coordinates - numpy.ndarray(bs, 3)",
  "global_rotation": "current robot orientation as quaternion - numpy.ndarray(bs, 4)",
  "globalyaw": "current robot yaw angle converted from quaternion - numpy.ndarray(bs,)",
  "gt_actions": "ground truth actions for supervised learning - numpy.ndarray(bs,)",
  "prev_actions": "previous step actions for temporal modeling - numpy.ndarray(bs,)",
  "rgb": "color image from robot camera - numpy.ndarray(bs, 256, 256, 3)",
  "depth": "depth image from robot camera - numpy.ndarray(bs, 256, 256, 1)"
}
```
如果需要修改训练数据所用的数据结构，可以根据[cma_dataset](../internnav/dataset/cma_dataset.py)自行实现。



## 评测数据
### metadata
评测数据的[metadata](../data/datasets/R2R_VLNCE_v1-3_for_challenge/gather_data/val_seen_gather_data.json)位于`data/datasets/R2R_VLNCE_v1-3_for_challenge/gather_data/val_seen_gather_data.json`中，其数据结构同[训练数据的metadata](#metadata)。

### 实际数据
通过GRUtopia框架获取得到的实际数据结构如下
```json
{
  "globalgps": "current robot position in world coordinates - numpy.ndarray(3,)",
  "globalrotation": "current robot orientation as quaternion - numpy.ndarray(4,)",
  "depth": "depth image from robot camera - numpy.ndarray(256,256,1)",
  "rgb": "color image from robot camera - numpy.ndarray(256,256,3)",
  "instruction": "natural language navigation command - str",
  "instruction_tokens": "tokenized version of instruction text - list[int]"
}
```

## GPU数据搬运

为了让数据能够在GPU上进行运算，方便模型处理，我们提供了[`batch_obs`](../internnav/evaluator/utils/models.py)函数将数据搬运到gpu上进行运算

## GRUtopia相关资源

为保证GRUtopia正确运行，需下载[相关资源](https://xxxxxx)。在eval时将资源路径加入环境变量，具体可参考[start_eval.sh](../scripts/eval/start_eval.sh)

## 数据下载

metadata: 位于 `/root/data/`，已经软链接到`data/`

训练数据：

场景文件：

GRUtopia资源文件：