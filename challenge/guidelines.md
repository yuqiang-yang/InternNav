# IROS 2025 Challenge Participation Guidelines (WIP)

This guide provides a step-by-step walkthrough for participating in the **IROS 2025 Challenge on Multimodal Robot Learning**—from setting up your environment and developing your model, to evaluating and submitting your results.

---

## 🔗 Useful Links
- 🔍 **Challenge Overview:**  
 [Challenge of Multimodal Robot Learning in InternUtopia and Real World](https://internrobotics.shlab.org.cn/challenge/2025/).

- 📖 **InternUtopia + InternNav Documentation:**  
 [Getting Started](https://internrobotics.github.io/user_guide/internutopia/get_started/index.html)

- 🚀 **Interactive Demo:**  
 [InternNav Model Inference Demo](https://huggingface.co/spaces/InternRobotics/InternNav-Eval-Demo)

---

## 🧩 Environment Setup

### Activate your virtual environment 
```bash
source .venv/{environment_name}/bin/activate
export PYTHONPATH="$(pwd):$PYTHONPATH"
```

### Pull our base Docker image
```bash
```

### Navigate to codebase
All code development and configuration should happen inside the `grnavigation` directory:

```bash
cd grnavigation
```

### Download the starter dataset (val_seen + val_unseen splits)
```bash
git lfs install
mkdir kujiale_data
# InteriorAgent scene usd
git clone https://huggingface.co/datasets/spatialverse/InteriorAgent kujiale_data/scene_data
# InteriorAgent train and val dataset
git clone https://huggingface.co/datasets/spatialverse/InteriorAgent_Nav kujiale_data/raw_data
# Latest InternData (required huggingface token to download, generate one from here https://huggingface.co/settings/tokens)
git clone -b v0.1-full https://huggingface.co/datasets/InternRobotics/InternData-N1 data
```

### Suggested Dataset Path
#### InternData-N1
```
data/ 
├── Embodiments/
├── scene_data/
│   ├── mp3d/
│   │   ├──17DRP5sb8fy/
│   │   ├── 1LXtFkjw3qL/
│   │   └── ...
├── vln_pe/
│   ├── raw_data/
│   │   ├── train/
│   │   ├── val_seen/
│   │   │   └── val_seen.json.gz
│   │   └── val_unseen/
│   │       └── val_unseen.json.gz
└── └── traj_data/
        ├── interior_agent/
        │   └── kujiale
        └── mp3d/
            └── trajectory_0/
                ├── data/
                ├── meta/
                └── videos/
```
#### Interior_data/
```
kujiale_data
├── scene_data      
│   ├── kujiale_xxxx/
│   └── ...
└── raw_data        
    ├── train/
    ├── val_seen/
    └── val_unseen/
```


### [Optional] Download the baseline model
```bash
# ddppo-models
mkdir -p checkpoints/ddppo-models
wget -P checkpoints/ddppo-models https://dl.fbaipublicfiles.com/habitat/data/baselines/v1/ddppo/ddppo-models/gibson-4plus-mp3d-train-val-test-resnet50.pth
# longclip-B
huggingface-cli download --include 'longclip-B.pt' --local-dir-use-symlinks False --resume-download Beichenzhang/LongCLIP-B --local-dir checkpoints/clip-long
# download r2r finetuned baseline checkpoints
git clone https://huggingface.co/InternRobotics/VLN-PE && mv VLN-PE/r2r checkpoints/
```

## 🛠️ Local Development & Testing

### Run the container
```bash
```
### Develop & test
- Implement your policy under `internnav/model` and add to `internav/agent`.
- We provide train and eval script. We provide baseline models as examples.
- Use our train script to train your model:
    ```bash
    ./scripts/train/start_train.sh --name train_rdp --model rdp
    ```
- Use our evaluation script for quick checks:
    ```bash
    ./scripts/eval/start_eval.sh --config scripts/eval/configs/challenge_cfg.py
    ```

## 📦 Packaging & Submission
### Make sure your trained weights & model is in `challenge_cfg.py`
```bash
# quick check 
./scripts/eval/start_eval_iros.sh --config scripts/eval/configs/challenge_cfg.py
```
### Build your submission image
```bash
docker build -t registry.cn-hangzhou.aliyuncs.com/yourteam/iros2025:dev .
```
### Push to the registry
```bash
docker push registry.cn-hangzhou.aliyuncs.com/yourteam/iros2025:dev
```
### Submit your image URL on Eval.AI

#### Submission Format

Create a JSON file with your Docker image URL and team information. The submission must follow this exact structure:

```json
{
    "url": "registry.cn-hangzhou.aliyuncs.com/yourteam/iros2025:dev",
    "team": {
        "name": "your-team-name",
        "members": [
            {
                "name": "John Doe",
                "affiliation": "University of Example",
                "email": "john.doe@example.com",
                "leader": true
            },
            {
                "name": "Jane Smith",
                "affiliation": "Example Research Lab",
                "email": "jane.smith@example.com",
                "leader": false
            }
        ]
    }
}
```

#### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `url` | string | Complete Docker registry URL for your submission image |
| `team.name` | string | Official team name for leaderboard display |
| `team.members` | array | List of all team members with their details |
| `members[].name` | string | Full name of team member |
| `members[].affiliation` | string | University or organization affiliation |
| `members[].email` | string | Valid contact email address |
| `members[].leader` | boolean | Team leader designation (exactly one must be `true`) |

For detailed submission guidelines and troubleshooting, refer to the official Eval.AI platform documentation.

## 📝 Official Evaluation Flow
### DSW Creation
- We use the AliCloud API to instantiate an instance from your image link.
- The system mounts the evaluation config + full dataset (val_seen, val_unseen, test).

### Evaluation Execution
- Via SSH + `screen`, we launch `challenge/start_eval_iros.sh`.
- A polling loop watches for result files.

### Results Collection
- Upon completion, metrics for each split are parsed and pushed to the [EvalAI](https://eval.ai/web/challenges/challenge-page/2627/overview) leaderboard.
- The released results are computed as a weighted sum of the test subsets from VLNPE-R2R (MP3D scenes) and Interior-Agent (Kujiale scenes), with a weighting ratio of 2:1.
