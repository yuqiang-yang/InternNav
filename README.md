<div align="center">

[![demo](assets/InternNav.gif "demo")](https://www.youtube.com/watch?v=fD0F1jIax5Y)

[![Gradio Demo](https://img.shields.io/badge/Gradio-Demo-orange?style=flat&logo=gradio)](http://123.57.187.96:55005/)
[![doc](https://img.shields.io/badge/Document-FFA500?logo=readthedocs&logoColor=white)](https://internrobotics.github.io/user_guide/internnav/index.html)
[![GitHub star chart](https://img.shields.io/github/stars/InternRobotics/InternNav?style=square)](https://github.com/InternRobotics/InternNav)
[![GitHub Issues](https://img.shields.io/github/issues/InternRobotics/InternNav)](https://github.com/InternRobotics/InternNav/issues)
<a href="https://cdn.vansin.top/taoyuan.jpg"><img src="https://img.shields.io/badge/WeChat-07C160?logo=wechat&logoColor=white" height="20" style="display:inline"></a>
[![Discord](https://img.shields.io/discord/1373946774439591996?logo=discord)](https://discord.gg/5jeaQHUj4B)

</div>

## 🏠 Introduction

InternNav is an All-in-one open-source toolbox for embodied navigation based on PyTorch, Habitat and Isaac Sim.

### Highlights
- Modular Support of the Entire Navigation System

We support modular customization and study of the entire navigation system, including vision-language navigation with discrete action space (VLN-CE), visual navigation (VN) given point/image/trajectory goals, and the whole VLN system with continuous trajectory outputs.

- Compatibility with Mainstream Simulation Platforms

The toolbox is compatible with different training and evaluation requirements, supporting different environments for the usage of mainstream simulation platforms such as Habitat and Isaac Sim.

- Comprehensive Datasets, Models and Benchmarks

The toolbox supports the most comprehensive 6 datasets \& benchmarks and 10+ popular baselines, including both mainstream and our established brand new ones.

- State of the Art

The toolbox supports the most advanced high-quality navigation dataset, InternData-N1, which includes 3k+ scenes and 830k VLN data covering diverse embodiments and scenes, and the first dual-system navigation foundation model with leading performance on all the benchmarks and zero-shot generalization capability in the real world, InternVLA-N1.

## 🔥 News

- [2025/07] We are hosting 🏆IROS 2025 Grand Challenge, stay tuned at [official website](https://internrobotics.shlab.org.cn/challenge/2025/).
- [2025/07] InternNav v0.1.0 released.

## 📋 Table of Contents
- [🏠 Introduction](#-introduction)
- [🔥 News](#-news)
- [📚 Getting Started](#-getting-started)
- [📦 Overview of Benchmark \& Model Zoo](#-benchmark-model-zoo)
- [🔧 Customization](#-customization)
- [👥 Contribute](#-contribute)
- [🔗 Citation](#-citation)
- [📄 License](#-license)
- [👏 Acknowledgements](#-acknowledgements)

## 📚 Getting Started

Please refer to the [documentation](https://internrobotics.github.io/user_guide/internnav/quick_start/index.html) for quick start with InternNav, from installation to training or evaluating supported models.

## 📦 Overview of Benchmark and Model Zoo

### Datasets \& Benchmarks

<table align="center">
  <tbody>
    <tr align="center" valign="bottom">
      <td>
         <b>System2 (VLN-CE)</b>
      </td>
      <td>
         <b>System1 (VN)</b>
      </td>
      <td>
         <b>Whole-system (VLN)</b>
      </td>
   </tr>
   <tr align="center" valign="top">
      <td>
         <ul>
            <li align="left"><a href="">VLN-CE R2R</a></li>
            <li align="left"><a href="">VLN-CE RxR</a></li>
         </ul>
      </td>
      <td>
         <ul>
            <li align="left"><a href="">Cluttered Envs</a></li>
            <li align="left"><a href="">GRScenes-100</a></li>
         </ul>
      </td>
      <td>
         <ul>
            <li align="left"><a href="">VLN-CE</a></li>
            <li align="left"><a href="">VLN-PE</a></li>
         </ul>
      </td>
   </tbody>
</table>

### Models

<table align="center">
  <tbody>
    <tr align="center" valign="bottom">
      <td>
         <b>System2 (VLN-CE)</b>
      </td>
      <td>
         <b>System1 (VN)</b>
      </td>
      <td>
         <b>Whole-system (VLN)</b>
      </td>
   </tr>
   <tr align="center" valign="top">
      <td>
         <ul>
            <li align="left"><a href="">StreamVLN</a></li>
            <li align="left"><a href="">InternVLA-N1 (S2)</a></li>
         </ul>
      </td>
      <td>
         <ul>
            <li align="left"><a href="">DD-PPO</a></li>
            <li align="left"><a href="">iPlanner</a></li>
            <li align="left"><a href="">ViPlanner</a></li>
            <li align="left"><a href="">GNM</a></li>
            <li align="left"><a href="">ViNT</a></li>
            <li align="left"><a href="">NoMad</a></li>
            <li align="left"><a href="">NavDP</a></li>
         </ul>
      </td>
      <td>
         <ul>
            <li align="left"><a href="">Seq2Seq</a></li>
            <li align="left"><a href="">CMA</a></li>
            <li align="left"><a href="">RDP</a></li>
            <li align="left"><a href="">InternVLA-N1</a></li>
         </ul>
      </td>
   </tbody>
</table>

**NOTE:**
- The detailed benchmark results will be updated in the next few days.
- VLN-CE RxR benchmark and StreamVLN will be supported soon.

## 🔧 Customization

Please refer to the [tutorial](https://internrobotics.github.io/user_guide/internnav/tutorials/index.html) for advanced usage of InternNav, including customization of datasets, models and experimental settings.

## 👥 Contribute

If you would like to contribute to InternNav, please check out our [contribution guide]().
For example, raising issues, fixing bugs in the framework, and adapting or adding new policies and data to the framework.

**Note:** We welcome the feedback of the model's zero-shot performance when deploying in your own environment. Please show us your results and offer us your future demands regarding the model's capability. We will select the most valuable ones and collaborate with users together to solve them in the next few months :)

## 🔗 Citation

If you find our work helpful, please cite:

```bibtex
@misc{internnav2025,
    title = {{InternNav: InternRobotics'} open platform for building generalized navigation foundation models},
    author = {InternNav Contributors},
    howpublished={\url{https://github.com/InternRobotics/InternNav}},
    year = {2025}
}
```

If you use the specific pretrained models and benchmarks, please kindly cite the original papers involved in our work. Related BibTex entries of our papers are provided below.

<details><summary>Related Work BibTex</summary>

```BibTex
@misc{internvla-n1,
    title = {{InternVLA-N1: An} Open Dual-System Navigation Foundation Model with Learned Latent Plans},
    author = {InternNav Team},
    year = {2025},
    booktitle={arXiv},
}
@inproceedings{vlnpe,
  title={Rethinking the Embodied Gap in Vision-and-Language Navigation: A Holistic Study of Physical and Visual Disparities},
  author={Wang, Liuyi and Xia, Xinyuan and Zhao, Hui and Wang, Hanqing and Wang, Tai and Chen, Yilun and Liu, Chengju and Chen, Qijun and Pang, Jiangmiao},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2025}
}
@misc{streamvln,
    title = {StreamVLN: Streaming Vision-and-Language Navigation via SlowFast Context Modeling},
    author = {Wei, Meng and Wan, Chenyang and Yu, Xiqian and Wang, Tai and Yang, Yuqiang and Mao, Xiaohan and Zhu, Chenming and Cai, Wenzhe and Wang, Hanqing and Chen, Yilun and Liu, Xihui and Pang, Jiangmiao},
    booktitle={arXiv},
    year = {2025}
}
@misc{navdp,
    title = {NavDP: Learning Sim-to-Real Navigation Diffusion Policy with Privileged Information Guidance},
    author = {Wenzhe Cai, Jiaqi Peng, Yuqiang Yang, Yujian Zhang, Meng Wei, Hanqing Wang, Yilun Chen, Tai Wang and Jiangmiao Pang},
    year = {2025},
    booktitle={arXiv},
}
```

</details>


## 📄 License

InternNav's codes are [MIT licensed](LICENSE). 
The open-sourced InternData-N1 data are under the <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License </a><a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/80x15.png" /></a>.
Other datasets like VLN-CE inherit their own distribution licenses.

## 👏 Acknowledgement

- [InternUtopia](https://github.com/InternRobotics/InternUtopia) (Previously `GRUtopia`): The closed-loop evaluation and GRScenes-100 data in this framework relies on the InternUtopia framework.
- [Diffusion Policy](https://github.com/real-stanford/diffusion_policy): Diffusion policy implementation.
- [LongCLIP](https://github.com/beichenzbc/Long-CLIP): Long-text CLIP model.
- [VLN-CE](https://github.com/jacobkrantz/VLN-CE): Vision-and-Language Navigation in Continuous Environments based on Habitat.
- [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL): The pretrained vision-language foundation model.
- [LeRobot](https://github.com/huggingface/lerobot): The data format used in this project largely follows the conventions of LeRobot.