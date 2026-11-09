"""
Full pipeline on the NICOL robot in CoppeliaSim, per instance:
  1. Perception: capture an image from NICOL's eye camera; OWLv2 locates task objects (left/center/right area)
  2. Planning:   LLM+MAP (LLM -> PDDL -> FMAP/BFWS) or the LLM-direct baseline
  3. Execution:  run the bimanual plan on NICOL and check task success
Requires CoppeliaSim running with NICOL/nicol.ttt loaded.
"""
import os, time, json, argparse, traceback
from log_utils import banner, log

##################################################################
# parameters
parser = argparse.ArgumentParser(
    description="Run full instances in CoppeliaSim: perception (OWLv2) -> planning (LLM+MAP or direct LLM) -> execution on NICOL.")
parser.add_argument("--task_name", type=str, default="ServeWater",
                    choices=["ServeWater", "ServeFruit", "StackBlock"], help="Task to run.")
parser.add_argument("--model_name", type=str, default="gpt-4o",
                    help="LLM name, e.g. gpt-4o, gpt-4.1, o1, deepseek-reasoner.")
parser.add_argument("--start", type=int, default=0, help="Index of the first instance.")
parser.add_argument("--task_nums", type=int, default=1, help="Number of instances to run (start .. start+N-1).")
parser.add_argument("--direct", action="store_true", help="Let the LLM plan directly instead of generating PDDL code.")
parser.add_argument("--use_bfws", type=str, default="true", choices=["true", "false"],
                    help="LLM+MAP only: also plan with the single-agent BFWS planner and keep the shorter of the "
                         "FMAP/BFWS plans (true, default), or plan with FMAP only (false; no BFWS binary needed).")
parser.add_argument("--no_owlv2", action="store_true",
                    help="Skip OWLv2 perception and use the ground-truth visual description from the task config.")
parser.add_argument("--load", action="store_true",
                    help="Load existing task configs from --configs_dir instead of sampling new scenes.")
parser.add_argument("--configs_dir", type=str, default="./task_configs",
                    help="Root of existing task configs used with --load (<configs_dir>/<task_name>/task_config_<i>.json).")
parser.add_argument("--output_dir", type=str, default=None,
                    help="Where configs, plans, images and CSVs of this run go. Default: ./results/main/<task_name>.")
parser.add_argument("--wandb", action="store_true", help="Log results to Weights & Biases.")
args = parser.parse_args()

task_name = args.task_name
model_name = args.model_name
use_pddl = not args.direct
use_bfws = args.use_bfws == "true"
use_owlv2 = not args.no_owlv2
log_wandb = args.wandb
##################################################################

# Heavy imports (simulator / NICOL stack / models) after argument parsing, so --help works without them
import wandb
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tasks import create_task, generate_visual_description
from owlv2_inference import owlvit_solve
from LLM.LMP import llm_pddl, llm_direct
from LLM.code_blocks import TMP_DIR, clean_tmp
from NICOL.nicol_controller import nicol_control

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Output folders of this run (created if missing, never wiped)
output_dir = args.output_dir or f"./results/main/{task_name}"
pddl_plan_path = os.path.join(output_dir, "pddl_plans")
direct_plan_path = os.path.join(output_dir, "direct_plans")
task_configs_path = os.path.join(output_dir, "task_configs")
image_path = os.path.join(output_dir, "images")
for path in [pddl_plan_path if use_pddl else direct_plan_path, task_configs_path, image_path]:
    os.makedirs(path, exist_ok=True)
load_configs_path = os.path.join(args.configs_dir, task_name)

task = create_task(task_name)
sim = task.sim

if log_wandb:
    wandb.init(
        project=f"LLM-MAP-{task_name}",
        config={
            "task_name": task_name,
            "task_nums": args.task_nums,
            "use_pddl": use_pddl,
            "use_bfws": use_bfws,
            "use_owlv2": use_owlv2,
            "model_name": model_name
        }
    )

mode = "pddl" if use_pddl else "direct"
method = "LLM+MAP" if use_pddl else "LLM-direct"
gen_csv = os.path.join(output_dir, f"{task_name}_{model_name}_{mode}_plan_gen.csv")
exe_csv = os.path.join(output_dir, f"{task_name}_{model_name}_{mode}_exe.csv")
results_gen = []
results_exe = []
banner(f"Perception -> planning ({method}) -> execution on NICOL", task=task_name, model=model_name,
       **({"planners": "FMAP + BFWS" if use_bfws else "FMAP only"} if use_pddl else {}),
       perception="OWLv2" if use_owlv2 else "ground truth (task config)",
       scenes="loaded from " + load_configs_path if args.load else "randomly sampled",
       instances=f"{args.start}..{args.start + args.task_nums - 1} ({args.task_nums})", output=output_dir)
for index in range(args.start, args.start + args.task_nums):
    print(f"\n{'-' * 80}")
    sim.startSimulation()
    try:
        ### Task configuration
        #################################################################
        config_path = os.path.join(task_configs_path, f"task_config_{index}.json")
        if args.load:
            with open(os.path.join(load_configs_path, f"task_config_{index}.json"), "r", encoding="utf-8") as f:
                config = json.load(f)
            task.reset(config)
        else:
            task.reset()
            config = task.config
        # keep a copy of the config used in this run next to its plans
        with open(config_path, "w") as f:
            json.dump(config, f)
        if log_wandb:
            wandb.save(config_path)
        # task.task_des is only set by reset() when sampling a new scene, so read it from the config
        task_des = config["task_des"]
        log(method, f"Instance {index}, scene type: {config['type']}")
        print(task_des)
        time.sleep(2)

        ### Scene understanding (Spatial)
        #################################################################
        log("Perception", "Capturing the scene from NICOL's eye camera...")
        task.get_image_from_eyes(index=index)
        if log_wandb:
            wandb.log({"task": index, "image": wandb.Image(task.image)})
        if use_owlv2:
            log("Perception", "OWLv2 is locating the task objects...")
            file_name = os.path.join(image_path, f"{task_name}_{index}_processed.png")
            directions, fig = owlvit_solve(Image.fromarray(task.image), task.text_queries, task.owl_threshold, file_name, 'owlv2')
            if log_wandb:
                wandb.log({"directions": fig})
            plt.close(fig)
            visual_description = generate_visual_description(task, directions)
        else:
            log("Perception", "Using the ground-truth visual description from the task config.")
            visual_description = config["task_des_visual"]
        task_description = task_des + visual_description

        ### Plan generation (Temporal)
        #################################################################
        log(method, "Generating the bimanual plan...")
        if use_pddl:
            code_file_path = os.path.join(pddl_plan_path, f"llm_pddl_{index}.py")
            plan_path = os.path.join(pddl_plan_path, f"pddl_plan_{index}.json")
            code_path = os.path.join(pddl_plan_path, f"code_string_{index}.json")
            clean_tmp()  # start each run with an empty ./tmp so no stale plan is picked up
            code_string, result_plan, llm_time_cost, pddl_time_cost = llm_pddl(model_name, task_name, task_description, code_file_path, temperature=0.1, use_bfws=use_bfws)
            if log_wandb:
                wandb.log({"task": index, "code_gen_time": llm_time_cost})
                wandb.log({"task": index, "plan_gen_time": pddl_time_cost})
                wandb.save(code_file_path, base_path=os.path.dirname(code_file_path))
            tmp_plan_path = os.path.join(TMP_DIR, "plan.json")
            plan_success = os.path.exists(tmp_plan_path)
            if plan_success:
                os.rename(tmp_plan_path, plan_path)
                if log_wandb:
                    wandb.save(plan_path, base_path=os.path.dirname(plan_path))
            clean_tmp()  # delete intermediate files of this run
            if code_string is not None:
                with open(code_path, "w", encoding="utf-8") as f:
                    f.write(code_string)
            log_gen = {
                "task_name": task_name,
                "task_index": index,
                "task_type": config["type"],
                "llm_time_cost": llm_time_cost,
                "pddl_time_cost": pddl_time_cost,
                "plan_success": plan_success,
                "is_FMAP": plan_success and "Partial-order plan from BFWS is the optimal" not in (result_plan or ""),
                "use_bfws": use_bfws,
            }
        else:
            plan_path = os.path.join(direct_plan_path, f"direct_plan_{model_name}_{index}.json")
            if os.path.exists(plan_path):
                os.remove(plan_path)  # so a stale plan from an earlier run is never executed
            result_plan, llm_time_cost = llm_direct(model_name, task_name, task_description, plan_path)
            plan_success = result_plan is not None and os.path.exists(plan_path)
            if log_wandb:
                wandb.log({"task": index, "plan_generation_time": llm_time_cost})
                if plan_success:
                    wandb.save(plan_path, base_path=os.path.dirname(plan_path))
            log_gen = {
                "task_name": task_name,
                "task_index": index,
                "task_type": config["type"],
                "llm_time_cost": llm_time_cost,
                "plan_success": plan_success,
            }
        log(method, f"Plan saved to {plan_path}" if plan_success else "No plan found.")

        ### Plan execution with NICOL
        #################################################################
        if plan_success:
            log("Execution", "Executing the bimanual plan on NICOL...")
            exe_start = time.time()
            success, finished_steps = nicol_control(plan_path, use_pddl, task)
            log("Execution", f"{'SUCCEEDED' if success else 'FAILED'} after {round(time.time() - exe_start, 2)}s "
                             f"(finished steps: {finished_steps}).")
        else:
            log("Execution", "Skipped: no plan to execute.")
            success, finished_steps = False, 0
        if log_wandb:
            wandb.log({"task": index, "Success": int(success)})
            wandb.log({"task": index, "Finished_steps": finished_steps})
        log_exe = {
            "task_name": task_name,
            "task_index": index,
            "task_type": config["type"],
            "success": success,
            "finished_steps": finished_steps
        }
        ##################################################################
        results_gen.append(log_gen)
        results_exe.append(log_exe)
        pd.DataFrame(results_gen).to_csv(gen_csv, index=False)
        pd.DataFrame(results_exe).to_csv(exe_csv, index=False)
    except Exception:
        # one failing instance should not stop the whole batch
        log(method, f"Instance {index} failed with an error:")
        traceback.print_exc()
    finally:
        sim.stopSimulation()
        time.sleep(4)
n_plans = sum(bool(r["plan_success"]) for r in results_gen)
n_success = sum(bool(r["success"]) for r in results_exe)
banner(f"{method} pipeline finished", plans=f"{n_plans}/{args.task_nums} instances",
       executions=f"{n_success}/{args.task_nums} succeeded", output=output_dir)
if log_wandb:
    wandb.finish()
