"""
Plan generation only (no simulator needed): read task configs with ground-truth visual descriptions
and generate bimanual plans with LLM+MAP (LLM -> PDDL -> FMAP/BFWS) or the LLM-direct baseline.
"""
import json
import os
import argparse
import pandas as pd
import wandb

from LLM.LMP import llm_pddl, llm_direct
from LLM.code_blocks import TMP_DIR, clean_tmp
from log_utils import banner, log

parser = argparse.ArgumentParser(description="Generate plans for LLM+MAP tasks.")
parser.add_argument("--task_name", type=str, default="ServeWater",
                    choices=["ServeWater", "ServeFruit", "StackBlock"], help="Task to plan for.")
parser.add_argument("--model_name", type=str, default="gpt-4o",
                    help="LLM name, e.g. gpt-4o, gpt-4.1, o1, deepseek-reasoner.")
parser.add_argument("--start", type=int, default=0, help="Index of the first task config.")
parser.add_argument("--task_nums", type=int, default=1, help="Number of task configs to run (start .. start+N-1).")
parser.add_argument("--direct", action="store_true", help="Let the LLM plan directly instead of generating PDDL code.")
parser.add_argument("--use_bfws", type=str, default="true", choices=["true", "false"],
                    help="LLM+MAP only: also plan with the single-agent BFWS planner and keep the shorter of the "
                         "FMAP/BFWS plans (true, default), or plan with FMAP only (false; no BFWS binary needed).")
parser.add_argument("--wandb", action="store_true", help="Log results to Weights & Biases.")
args = parser.parse_args()

task_name = args.task_name
model_name = args.model_name
use_pddl = not args.direct
use_bfws = args.use_bfws == "true"
log_wandb = args.wandb
method = "LLM+MAP" if use_pddl else "LLM-direct"
indices = list(range(args.start, args.start + args.task_nums))

task_configs_path = f"./task_configs/{task_name}"
direct_plan_path = f"./results/direct_plans/{task_name}"
pddl_plan_path = f"./results/pddl_plans/{task_name}"
os.makedirs(pddl_plan_path if use_pddl else direct_plan_path, exist_ok=True)
csv_path = f"./results/{task_name}_{model_name}_{'pddl' if use_pddl else 'direct'}_plan_gen.csv"

if log_wandb:
    wandb.init(
        project=f"LLM-MAP-{task_name}",
        config={
            "task_name": task_name,
            "task_nums": len(indices),
            "model_name": model_name,
            "use_pddl": use_pddl,
            "use_bfws": use_bfws,
            "plan_generation": True,
            "plan_validation": False,
        }
    )

banner(f"Plan generation with {method}", task=task_name, model=model_name,
       **({"planners": "FMAP + BFWS" if use_bfws else "FMAP only"} if use_pddl else {}),
       instances=f"{indices[0]}..{indices[-1]} ({len(indices)})" if indices else "none")
results = []
for count, index in enumerate(indices, 1):
    config_path = task_configs_path + f"/task_config_{index}.json"
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    print(f"\n{'-' * 80}")
    log(method, f"Instance {index} ({count}/{len(indices)}), scene type: {config['type']}")
    task_description = config["task_des"] + config["task_des_visual"]
    clean_tmp()  # start each run with an empty ./tmp so no stale plan is picked up

    if use_pddl:
        code_file_path = pddl_plan_path + f'/llm_pddl_{index}.py'
        plan_path = pddl_plan_path + f"/pddl_plan_{index}.json"
        code_path = pddl_plan_path + f"/code_string_{index}.json"
        code_string, result_plan, llm_time_cost, pddl_time_cost = llm_pddl(model_name, task_name, task_description, code_file_path, temperature=0.1, use_bfws=use_bfws)
        if log_wandb:
            wandb.log({"task": index, "code_gen_time": llm_time_cost})
            wandb.log({"task": index, "plan_gen_time": pddl_time_cost})
            wandb.save(code_file_path, base_path=os.path.dirname(code_file_path))
        tmp_plan_path = os.path.join(TMP_DIR, "plan.json")
        plan_success = os.path.exists(tmp_plan_path)
        is_fmap = plan_success and "Partial-order plan from BFWS is the optimal" not in (result_plan or "")
        if plan_success:
            os.rename(tmp_plan_path, plan_path)
            if log_wandb:
                wandb.save(plan_path, base_path=os.path.dirname(plan_path))
            log(method, f"Plan saved to {plan_path} (selected planner: {'FMAP' if is_fmap else 'BFWS'})")
        else:
            log(method, "No plan found for this instance.")
        if code_string is not None:
            with open(code_path, "w", encoding="utf-8") as f:
                f.write(code_string)
        record = {
            "task_name": task_name,
            "task_index": index,
            "task_type": config["type"],
            "llm_time_cost": llm_time_cost,
            "pddl_time_cost": pddl_time_cost,
            "plan_success": plan_success,
            "is_FMAP": is_fmap,
            "use_bfws": use_bfws,
            }
    else:
        plan_path = direct_plan_path + f"/direct_plan_{model_name}_{index}.json"
        if os.path.exists(plan_path):
            os.remove(plan_path)  # never report a stale plan from an earlier run
        result_plan, llm_time_cost = llm_direct(model_name, task_name, task_description, plan_path)
        plan_success = result_plan is not None
        if log_wandb:
            wandb.log({"task": index, "plan_generation_time": llm_time_cost})
            if plan_success:
                wandb.save(plan_path, base_path=os.path.dirname(plan_path))
        if plan_success:
            log(method, f"Plan saved to {plan_path}")
        record = {
            "task_name": task_name,
            "task_index": index,
            "task_type": config["type"],
            "llm_time_cost": llm_time_cost,
            "plan_success": plan_success,
            }
    clean_tmp()  # delete intermediate files of this run

    results.append(record)
    pd.DataFrame(results).to_csv(csv_path, index=False)

n_success = sum(r["plan_success"] for r in results)
banner(f"Plan generation with {method} finished", plans=f"{n_success}/{len(results)} instances", csv=csv_path)
if log_wandb:
    wandb.finish()
