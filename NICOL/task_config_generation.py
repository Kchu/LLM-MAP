import wandb
import os, sys, time
import json
import argparse

# This script lives in NICOL/ but imports tasks.py from the repo root.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

##################################################################
# parameters
parser = argparse.ArgumentParser(description="Generate (or reload) task configurations in CoppeliaSim and capture scene images.")
parser.add_argument("--task_name", type=str, default="ServeWater",
                    choices=["ServeWater", "ServeFruit", "StackBlock"], help="Task to generate configs for.")
parser.add_argument("--start", type=int, default=0, help="Index of the first task config.")
parser.add_argument("--task_nums", type=int, default=50, help="Number of task configs to generate (start .. start+N-1).")
parser.add_argument("--configs_dir", type=str, default=os.path.join(REPO_ROOT, "task_configs"),
                    help="Root folder of task configs; files go to <configs_dir>/<task_name>/task_config_<i>.json.")
parser.add_argument("--load", action="store_true", help="Reload existing configs instead of sampling new ones.")
parser.add_argument("--wandb", action="store_true", help="Log results to Weights & Biases.")
args = parser.parse_args()

# Heavy imports (simulator / NICOL stack) after argument parsing, so --help works without them
from tasks import create_task

task_name = args.task_name
log_wandb = args.wandb
task_configs_path = os.path.join(args.configs_dir, task_name)
os.makedirs(task_configs_path, exist_ok=True)

task = create_task(task_name)
sim = task.sim

if log_wandb:
    wandb.init(
        project=f"LLM-MAP-{task_name}",
        config={
            "task_name": task_name,
            "task_nums": args.task_nums,
            "use_pddl": False,
            "model_name": None,
            "task_config_gen": True
        }
    )

for i in range(args.start, args.start + args.task_nums):
    print(f"*"*80)
    sim.startSimulation()
    time.sleep(2)
    config_path = os.path.join(task_configs_path, f"task_config_{i}.json")
    if args.load:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        task.reset(config)
    else:
        task.reset()
        ## store the task configurations
        with open(config_path, "w") as f:
            json.dump(task.config, f)
        if log_wandb:
            wandb.save(config_path)
    task.get_image_from_eyes(i)
    time.sleep(2)
    print('\n Task: ', i, "type", task.config["type"])
    sim.stopSimulation()
    time.sleep(2)
if log_wandb:
    wandb.finish()
