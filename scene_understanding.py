from owlv2_inference import owlvit_solve
import json
import argparse
import pandas as pd
from PIL import Image
import os
import matplotlib.pyplot as plt

os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser(description="Evaluate OWLv2 scene understanding against ground-truth task configs.")
parser.add_argument("--task_name", type=str, default="ServeFruit",
                    choices=["ServeWater", "ServeFruit", "StackBlock"], help="Task to evaluate.")
parser.add_argument("--task_nums", type=int, default=1, help="Evaluate images/configs 0 .. N-1.")
parser.add_argument("--image_dir", type=str, default="./images", help="Folder with the scene images.")
parser.add_argument("--image_prefix", type=str, default=None,
                    help="Image file prefix; images are <image_dir>/<prefix>_<i>.png. Defaults to the task name.")
parser.add_argument("--configs_dir", type=str, default="./task_configs",
                    help="Root folder of task configs (<configs_dir>/<task_name>/task_config_<i>.json).")
parser.add_argument("--output_dir", type=str, default="./results", help="Where the result CSV is written.")
args = parser.parse_args()

task_name = args.task_name
image_prefix = args.image_prefix or task_name
task_configs_path = os.path.join(args.configs_dir, task_name)
# OWLv2 visualizations go to the output folder, so images/ keeps only the original scene images
vis_dir = os.path.join(args.output_dir, "owlv2_images")
os.makedirs(vis_dir, exist_ok=True)

# text_queries[j] is the OWLv2 query whose ground truth is config["area"][gt_keys[j]]
if task_name == "ServeFruit":
    text_queries = ["red bowl", "apple", "banana"]
    gt_keys = ["bowl", "apple", "banana"]
    owl_threshold = 0.04
elif task_name == "StackBlock":
    text_queries = ["human face", "red cube", "green cube", "blue cube", "yellow cube", "purple cube", "black cube"]
    gt_keys = ["human", "red", "green", "blue", "yellow", "purple", "black"]
    owl_threshold = 0.03
elif task_name == "ServeWater":
    text_queries = ["human face", "blue cup", "yellow cup", "brown box"]
    gt_keys = ["human", "blue cup", "yellow cup", "brown box"]
    owl_threshold = 0.01

vlm_results = []
for i in range(args.task_nums):
    path = os.path.join(args.image_dir, f"{image_prefix}_{i}.png")
    file_name = os.path.join(vis_dir, f"{image_prefix}_{i}_owlv2.png")
    print('\n Task: ', i)
    areas, fig = owlvit_solve(Image.open(path), text_queries, owl_threshold, file_name, 'owlv2')
    plt.close(fig)

    config_path = os.path.join(task_configs_path, f"task_config_{i}.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    gt_areas = config["area"]
    print("gt_areas", gt_areas)
    correct_matches = 0
    mismatches = {}
    for j in range(len(gt_keys)):
        if areas[text_queries[j]] == gt_areas[gt_keys[j]]:
            correct_matches += 1
        else:
            mismatches[text_queries[j]] = (areas[text_queries[j]], gt_areas[gt_keys[j]])
    match_rate = correct_matches / len(gt_keys)
    print(f"Match rate: {match_rate*100:.2f}%")
    if match_rate < 1:
        print("mismatch items are: ", mismatches)

    vlm_results.append({
        "task_name": task_name,
        "task_index": i,
        "task_type": config["type"],
        "match_rate": match_rate,
        "mismatch_items": mismatches
    })
    df = pd.DataFrame(vlm_results)
    df.to_csv(os.path.join(args.output_dir, f"{task_name}_scene_understanding_owlv2.csv"), index=False)
