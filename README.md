# LLM-MAP

Official implementation of [LLM+MAP: Bimanual Robot Task Planning using Large Language Models and Planning Domain Definition Language](https://arxiv.org/abs/2503.17309).
The code runs LLM+MAP on the bimanual humanoid robot NICOL in CoppeliaSim and includes the evaluation scripts.

## 🎥 Demo Video

[![LLM+MAP demo](https://markdown-videos-api.jorgenkh.no/url?url=https://youtu.be/i2j1WyNaBa8)](https://youtu.be/i2j1WyNaBa8)

> [!NOTE]
> **What can be run.** The NICOL API (`nicol_api`) is not open-sourced yet.
> - **Runnable:** plan generation (`plan_generation.py`) works without the robot or simulator. So do `ablation_study.py` and `scene_understanding.py`.
> - **Not runnable outside our setup:** the scripts that control NICOL in CoppeliaSim: `main.py`, `plan_validation.py` and `NICOL/task_config_generation.py`. They are included for reference.

## Contents

- [Tasks](#tasks)
- [Repository structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Outputs](#outputs)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Tasks

| Task | Goal | Skills | Configs |
|---|---|---|---|
| `ServeWater` | Pour the water from the blue cup into the yellow cup. Serve the yellow cup to the human and put the blue cup into the box. | grasp, move_to, move_above, pour, release | 100 |
| `ServeFruit` | Put the apple and the banana into a large bowl. Bring the bowl to the human with both hands. | grasp, move_to, move_above, place, push, release, co_hold, co_move_to | 100 |
| `StackBlock` | Stack the selected colored blocks into two piles on the tray in front of the human. | grasp, move_to, move_above, place, release | 100 |

Each task config (`task_configs/<task>/task_config_<i>.json`) holds a randomly sampled scene:
- the object poses,
- the ground-truth table areas of the objects (left / overlap / right),
- the task description and the ground-truth visual description.

`plan_generation.py` uses these configs directly, so no simulator is needed.

## Repository structure

```
├── plan_generation.py          # Plan generation from the task configs (runnable, no simulator needed)
├── ablation_study.py           # LLM+P (single-agent BFWS only) vs. LLM+MAP on the saved LLM code (runnable)
├── scene_understanding.py      # Evaluate OWLv2 object-area recognition on images/ (runnable)
├── main.py                     # Full run on NICOL: perception → planning → execution (needs NICOL API)
├── plan_validation.py          # Execute saved plans on NICOL and measure success (needs NICOL API)
├── owlv2_inference.py          # OWLv2 detection + mapping of detections to table areas
├── tasks.py                    # Task classes for the simulator (needs NICOL API)
├── log_utils.py                # Console logging helpers
├── LLM/
│   ├── LMP.py                  # LLM calls: llm_pddl (LLM+MAP) and llm_direct (LLM-direct baseline)
│   ├── prompts.py              # System prompts, rules and per-task skill descriptions
│   └── code_blocks.py          # PDDL domain, problem template, actions and FMAP/BFWS solver code
├── NICOL/
│   ├── nicol.ttt               # CoppeliaSim scene with the NICOL robot
│   ├── ServeWater.ttm, ServeFruit.ttm, StackBlock.ttm   # Task models
│   ├── nicol_controller.py     # Runs a two-hand plan step by step
│   ├── nicol_executor.py       # Robot skills via the NICOL API
│   ├── utils.py                # Plan parsing into per-hand commands
│   └── task_config_generation.py   # Sample new task configs in the simulator (needs NICOL API)
├── task_configs/<task>/        # 100 scene configs per task
└── images/                     # Example scene images from NICOL's eye camera
```

## Installation

**1. Python environment** (tested with Python 3.10):

```bash
conda create -n llm_map python=3.10 && conda activate llm_map
pip install -r requirements.txt
```

**2. FMAP.** It is installed with `up-fmap`, but it needs a Java runtime on your `PATH` (tested with OpenJDK 17).

**3. BFWS planner** (optional: skip this if you plan with FMAP only, `--use_bfws false`).
1. Build the `ff-version` of [BFWS](https://github.com/nirlipo/BFWS-public) against [LAPKT](https://github.com/LAPKT-dev/LAPKT-public), following their instructions.
2. Tell the code where the binary is:

```bash
export BFWS_BIN=/path/to/BFWS-public/ff-version/bfws   # or put `bfws` on your PATH
```

**4. LLM API keys.** Keys are only read from environment variables:

```bash
export OPENAI_API_KEY=...            # OpenAI models (gpt-4o, gpt-4.1, o1, ...)
export OPENAI_BASE_URL=...           # optional: any OpenAI-compatible endpoint
export DEEPSEEK_API_KEY=...          # optional: deepseek-* models
```

## Usage

Run all scripts from the repository root, because paths such as `./task_configs` and `./tmp` are relative to it.
Every script supports `--help`.
Add `--wandb` to log to Weights & Biases (project `LLM-MAP-<task>`).

**Plan generation** (runnable):

```bash
python plan_generation.py --task_name ServeFruit --model_name gpt-4.1 --task_nums 10                    # LLM+MAP (FMAP + BFWS)
python plan_generation.py --task_name ServeFruit --model_name gpt-4.1 --task_nums 10 --use_bfws false   # LLM+MAP (FMAP only)
python plan_generation.py --task_name ServeFruit --model_name gpt-4.1 --task_nums 10 --direct           # LLM-direct baseline
```

`--use_bfws` (default `true`) controls the planners used for LLM+MAP:
- **`true`:** the PDDL problem is solved by both the multi-agent FMAP planner and the single-agent BFWS planner, and the plan with fewer parallel steps is kept.
- **`false`:** only FMAP is used, so the BFWS binary is not needed.

`main.py` accepts the same flag.

Each run prints its progress, for example:

```
[12:00:18] [LLM+MAP] Instance 0 (1/1), scene type: ['right', 'right', 'right']
[12:00:18] [LLM+MAP] Trial 1/4: gpt-4.1 is generating the PDDL problem code...
[12:00:23] [LLM+MAP] Solving the generated PDDL problem with FMAP (multi-agent) and BFWS (single-agent)...
[12:00:25] [LLM+MAP] Plan saved to ./results/pddl_plans/ServeFruit/pddl_plan_0.json (selected planner: FMAP)
```

**Evaluation scripts** (runnable):

```bash
python ablation_study.py --task_name ServeFruit --task_nums 10        # LLM+P vs. LLM+MAP; run plan_generation.py first
python scene_understanding.py --task_name StackBlock --task_nums 1    # OWLv2 area recognition on images/
```

**Running on NICOL** (requires the NICOL API, not yet public). Start CoppeliaSim with `NICOL/nicol.ttt` loaded first.

```bash
python plan_validation.py --task_name ServeFruit --model_name gpt-4.1 --task_nums 10   # execute generated plans
python main.py --task_name ServeWater --model_name gpt-4.1 --task_nums 5               # perception → planning → execution
python NICOL/task_config_generation.py --task_name ServeWater --task_nums 100          # sample new task configs
```

## Outputs

| Path | Content |
|---|---|
| `results/pddl_plans/<task>/llm_pddl_<i>.py` | Full planning script: fixed code blocks + the LLM-generated problem |
| `results/pddl_plans/<task>/code_string_<i>.json` | LLM-generated problem code only |
| `results/pddl_plans/<task>/pddl_plan_<i>.json` | LLM+MAP plan |
| `results/direct_plans/<task>/direct_plan_<model>_<i>.json` | LLM-direct plan |
| `results/llm+p/<task>/` | Ablation: single-agent scripts and plans |
| `results/owlv2_images/` | OWLv2 detections drawn by `scene_understanding.py` |
| `results/<task>_<model>_<pddl\|direct>_plan_gen.csv` | Planning success and LLM/planner time per instance |
| `results/<task>_<model>_<pddl\|direct>_val.csv` | Execution success per instance (`plan_validation.py`) |

Plans are step-aligned JSON: `{"Left": [...], "Right": [...]}`. Both lists always have the same length.
Step *k* of both hands is executed at the same time, and `"wait"` keeps a hand still.

Intermediate planner files (PDDL files, BFWS output) are written to `./tmp/` and deleted after each instance.

## Troubleshooting

- **`BFWS planner 'bfws' not found`**: set `BFWS_BIN` (see [Installation](#installation)), or plan with FMAP only using `--use_bfws false`. The check runs before any LLM call.
- **`Set OPENAI_API_KEY ...`**: export the API key for the model you use.
- **`UPNoRequestedEngineAvailableException`** for FMAP: the generated scripts run with the same Python as the calling script. Make sure that environment has `up-fmap` installed and Java on its `PATH`.
- **`ModuleNotFoundError: nicol_api`**: the script needs the NICOL API (see the note at the top).
- **No plan found after 4 trials**: the LLM-generated problem may be unsolvable, for example a goal that cannot be reached from the initial state. Inspect `results/pddl_plans/<task>/llm_pddl_<i>.py` and run it directly with `python` to see the planners' output.

## License

This project is released under the [MIT License](LICENSE).
