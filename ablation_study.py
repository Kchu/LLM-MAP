import json
import wandb
import os, sys
import argparse
import pandas as pd
from LLM.code_blocks import *
from LLM.LMP import write_plan_file
import subprocess
import time

##################################################################
# parameters
parser = argparse.ArgumentParser(
    description="Ablation study: re-run the saved LLM code strings with the single-agent BFWS solver (LLM+P) "
                "and the multi-agent solver (LLM+MAP), and compare.")
parser.add_argument("--task_name", type=str, default="ServeFruit",
                    choices=["ServeWater", "ServeFruit", "StackBlock"], help="Task to evaluate.")
parser.add_argument("--model_name", type=str, default="gpt-4o",
                    help="LLM that produced the code strings (only recorded in wandb).")
parser.add_argument("--task_nums", type=int, default=1, help="Evaluate code_string_0 .. N-1.")
parser.add_argument("--output_dir", type=str, default="./results", help="Where the result CSVs are written.")
parser.add_argument("--wandb", action="store_true", help="Log results to Weights & Biases.")
args = parser.parse_args()

task_name = args.task_name
model_name = args.model_name
task_nums = args.task_nums
log_wandb = args.wandb
##################################################################

main_path = "./results"
pddl_plan_path = f"{main_path}/pddl_plans/{task_name}"       # code strings saved by plan_generation.py
ablation_study_path = f"{main_path}/llm+p/{task_name}"       # generated LLM+P scripts and plans
os.makedirs(ablation_study_path, exist_ok=True)
check_bfws()
os.makedirs(args.output_dir, exist_ok=True)


def build_plan_file(solver_definition, code_string, plan_file_path):
    write_plan_file(plan_file_path, domain_definition, ma_problem_initialization, plan2action,
                    code_string, choose_actions(task_name), solver_definition)


def run_plan_file(plan_file_path):
    # sys.executable: run the generated script with the same Python env as this script
    result = subprocess.run([sys.executable, plan_file_path], capture_output=True, text=True, check=True)
    return result.stdout.strip()


if log_wandb:
    wandb.init(
        project=f"LLM-MAP-{task_name}",
        config={
            "task_name": task_name,
            "task_nums": task_nums,
            "model_name": model_name,
            "use_pddl": True,
            "plan_generation": True,
            "plan_validation": False,
            "ablation_study": True
        }
    )
print(f"*"*100)
results_llm_p = []
results_llm_map = []
for index in range(task_nums):
    print("Task: ", index)
    code_string_path = pddl_plan_path + f"/code_string_{index}.json"
    with open(code_string_path, "r", encoding="utf-8") as f:
        code_string = f.read()

    #### LLM+P (single-agent BFWS)
    plan_path = ablation_study_path + f"/pddl_plan_{index}_sa.json"
    plan_file_path = ablation_study_path + f"/llm+P_plan_{index}.py"
    build_plan_file(sa_solver_definition, code_string, plan_file_path)
    clean_tmp()  # start each run with an empty ./tmp so no stale plan is picked up
    try:
        print("Looking for BFWS plan...")
        print("\n Plan: \n", run_plan_file(plan_file_path))
    except subprocess.CalledProcessError as e:
        print(f"Error executing the code you generated: {e.stderr.strip()}")
    plan_steps = None
    tmp_plan_path = os.path.join(TMP_DIR, "plan_sa.json")
    if os.path.exists(tmp_plan_path):
        with open(tmp_plan_path, "r", encoding="utf-8") as f:
            plan_steps = json.load(f)
        os.rename(tmp_plan_path, plan_path)
        if log_wandb:
            wandb.save(plan_path, base_path=os.path.dirname(plan_path))
    else:
        print("No plan found!")
    clean_tmp()  # delete intermediate files of the LLM+P run
    print(f"*"*60)
    log_llm_p = {
        "task_name": task_name,
        "task_index": index,
        "plan_steps": len(plan_steps) if plan_steps is not None else None,
        "plan_success": plan_steps is not None,
    }

    #### LLM+MAP (multi-agent FMAP + BFWS)
    plan_path = pddl_plan_path + f"/pddl_plan_{index}.json"
    plan_file_path = pddl_plan_path + f"/llm_pddl_{index}.py"
    build_plan_file(map_solver_definition, code_string, plan_file_path)
    clean_tmp()
    pddl_time_cost = None
    try:
        print("Looking for FMAP plan... ")
        pddl_start = time.time()
        output = run_plan_file(plan_file_path)
        pddl_time_cost = round(time.time() - pddl_start, 2)
        print(output)
    except subprocess.CalledProcessError as e:
        print(f"Error executing the code you generated: {e.stderr.strip()}")
    plan_data = None
    tmp_plan_path = os.path.join(TMP_DIR, "plan.json")
    if os.path.exists(tmp_plan_path):
        with open(tmp_plan_path, "r", encoding="utf-8") as f:
            plan_data = json.load(f)
        os.rename(tmp_plan_path, plan_path)
        print(f"*"*60)
    else:
        print("No plan found!")
    clean_tmp()  # delete intermediate files of the LLM+MAP run
    log_llm_map = {
        "task_name": task_name,
        "task_index": index,
        "plan_steps": len(plan_data["Left"]) if plan_data is not None else None,
        "plan_success": plan_data is not None,
        "pddl_time_cost": pddl_time_cost
    }

##################################################################
    results_llm_p.append(log_llm_p)
    results_llm_map.append(log_llm_map)
    pd.DataFrame(results_llm_p).to_csv(os.path.join(args.output_dir, f"LLM+P_{task_name}_plan_gen.csv"), index=False)
    pd.DataFrame(results_llm_map).to_csv(os.path.join(args.output_dir, f"LLM+MAP_{task_name}_plan_gen.csv"), index=False)
print(f"*"*100)
if log_wandb:
    wandb.finish()
