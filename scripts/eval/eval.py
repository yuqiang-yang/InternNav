import sys
sys.path.append('.')

from grnavigation.configs.model.seq2seq import seq2seq_eval_cfg
from grnavigation.configs.model.cma import cma_eval_cfg
from grnavigation.configs.model.rdp import rdp_eval_cfg
from grnavigation.configs.evaluator.default_config import get_config
from grnavigation.evaluator import Evaluator
from grnavigation.evaluator.utils.config import Config  # Ensure Config is imported
import argparse
import importlib.util



# This file is the main file


def parse_args():
    parser = argparse.ArgumentParser(description='评估导航模型')
    
    parser.add_argument(
        "--config",
        type=str,
        default='scripts/eval/configs/h1_cma_cfg.py',
        help='eval config file path, e.g. scripts/eval/configs/h1_cma_cfg.py',
    )


    
    return parser.parse_args()

def load_eval_cfg(config_path, attr_name='eval_cfg'):
    spec = importlib.util.spec_from_file_location("eval_config_module", config_path)
    config_module = importlib.util.module_from_spec(spec)
    sys.modules["eval_config_module"] = config_module
    spec.loader.exec_module(config_module)
    return getattr(config_module, attr_name)



def main():
    args = parse_args()
    

    
    # cfg_setting = Config(
    #     task_name=args.task_name,
    #     robot_name=args.robot_name,
    #     model_name=args.model_name,
    #     model_setting=model_setting,
    #     ckpt_path=args.ckpt_path,
    #     base_data_dir=args.base_data_dir,
    #     sim_num=args.sim_num,
    #     env_num=args.env_num,
    #     use_distributed=not args.not_use_distributed,
    #     headless=not args.not_headless,
    #     port=args.port,
    # )

    evaluator_cfg = load_eval_cfg(args.config, attr_name='eval_cfg')
    cfg = get_config(evaluator_cfg)
    print(cfg)
    evaluator = Evaluator.init(cfg)
    print(type(evaluator))
    # print(evaluator.__dict__)
    # import pdb; pdb.set_trace()
    evaluator.eval()


if __name__ == '__main__':
    main()
