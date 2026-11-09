"""
Plan validation: execute previously generated plans (from plan_generation.py) on the NICOL robot in
CoppeliaSim and record execution success. Requires CoppeliaSim running with NICOL/nicol.ttt loaded.
"""
import wandb
import os, time
import json
import argparse
import traceback
import pandas as pd
from log_utils import banner, log


##################################################################
# parameters
parser = argparse.ArgumentParser(description="Execute generated plans on NICOL in CoppeliaSim and record success.")
parser.add_argument("--task_name", type=str, default="ServeWater",
                    choices=["ServeWater", "ServeFruit", "StackBlock"], help="Task to validate.")
parser.add_argument("--model_name", type=str, default="gpt-4o",
                    help="LLM that produced the plans (used for direct-plan file names and the output CSV name).")
parser.add_argument("--task_nums", type=int, default=1, help="Validate task_config_0 .. N-1 (ignored if --indices is given).")
parser.add_argument("--indices", type=int, nargs="+", default=None,
                    help="Validate only these task indices, e.g. --indices 25 35 48.")
parser.add_argument("--direct", action="store_true", help="Validate direct LLM plans instead of PDDL plans.")
parser.add_argument("--wandb", action="store_true", help="Log results to Weights & Biases.")
args = parser.parse_args()

# Heavy imports (simulator / NICOL stack) after argument parsing, so --help works without them
from tasks import create_task
from NICOL.nicol_controller import nicol_control
from NICOL.utils import *

task_name = args.task_name
model_name = args.model_name
pddl = not args.direct
log_wandb = args.wandb
indices = args.indices if args.indices is not None else list(range(args.task_nums))
##################################################################

# Same folder layout as plan_generation.py
task_configs_path = f"./task_configs/{task_name}"
pddl_plan_path = f"./results/pddl_plans/{task_name}"
direct_plan_path = f"./results/direct_plans/{task_name}"
plan_path_root = pddl_plan_path if pddl else direct_plan_path
method = "LLM+MAP" if pddl else "LLM-direct"
os.makedirs("./results", exist_ok=True)
csv_path = f"./results/{task_name}_{model_name}_{'pddl' if pddl else 'direct'}_val.csv"

task = create_task(task_name)
sim = task.sim

if log_wandb:
    wandb.init(
        project=f"LLM-MAP-{task_name}",
        config={
            "task_name": task_name,
            "task_nums": len(indices),
            "model_name": model_name,
            "plan_pddl": pddl,
            "plan_generation": False,
            "plan_validation": True,
        }
    )

tasks_info = []
banner(f"Plan validation of {method} plans on NICOL (CoppeliaSim)", task=task_name, model=model_name,
       instances=f"{indices} ({len(indices)})")
for count, i in enumerate(indices, 1):
    print(f"\n{'-' * 80}")
    if pddl:
        plan_path = plan_path_root + f"/pddl_plan_{i}.json"
    else:
        plan_path = plan_path_root + f"/direct_plan_{model_name}_{i}.json"
    if not os.path.exists(plan_path):
        log(method, f"Instance {i} ({count}/{len(indices)}): no plan at {plan_path}, skipping (counted as failure).")
        tasks_info.append({"task_name": task_name, "task_index": i, "task_type": None, "plan_steps": None,
                           "success": False, "number_of_waits": None, "failed starting from step": 0})
        pd.DataFrame(tasks_info).to_csv(csv_path, index=False)
        continue

    sim.startSimulation()
    try:
        config_path = task_configs_path + f"/task_config_{i}.json"
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        log(method, f"Instance {i} ({count}/{len(indices)}), scene type: {config['type']}")
        log(method, "Resetting the scene from the task config...")
        task.reset(config)
        time.sleep(1)
        if log_wandb:
            wandb.save(config_path)
        time.sleep(2)

        with open(plan_path, "r", encoding="utf-8") as file:
            plan = json.load(file)
        left_dict, right_dict = plan2dict(plan["Left"]), plan2dict(plan["Right"])
        if pddl:
            left_commands, left_paras = command_parser_pddl(left_dict)
            right_commands, right_paras = command_parser_pddl(right_dict)
        else:
            left_commands, left_paras = command_parser_direct(left_dict)
            right_commands, right_paras = command_parser_direct(right_dict)
        num_of_waits = left_commands.count("wait") + right_commands.count("wait")
        log(method, f"Executing the {len(left_commands)}-step bimanual plan on NICOL...")
        start = time.time()
        success, finished_steps = nicol_control(plan_path, pddl, task)
        log(method, f"Execution {'SUCCEEDED' if success else 'FAILED'} after {round(time.time() - start, 2)}s "
                    f"(finished steps: {finished_steps}/{len(left_commands)}).")
        if log_wandb:
            wandb.log({"task": i, "Success": int(success)})
            wandb.log({"task": i, "Finished_steps": finished_steps})
            wandb.log({"task": i, "Number_of_waits": num_of_waits})
        ##################################################################
        tasks_info.append({
            "task_name": task_name,
            "task_index": i,
            "task_type": config["type"],
            "plan_steps": len(left_commands),
            "success": success,
            "number_of_waits": num_of_waits,
            "failed starting from step": finished_steps
        })
        pd.DataFrame(tasks_info).to_csv(csv_path, index=False)
    except Exception:
        # one failing instance should not stop the whole batch
        log(method, f"Instance {i} failed with an error:")
        traceback.print_exc()
    finally:
        sim.stopSimulation()
        time.sleep(4)

n_success = sum(bool(t["success"]) for t in tasks_info)
banner(f"Plan validation of {method} plans finished", success=f"{n_success}/{len(indices)} instances", csv=csv_path)
if log_wandb:
    wandb.finish()
