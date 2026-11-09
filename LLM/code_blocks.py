import os, shutil

# Scratch folder (relative to the repo root, where all scripts are run from) for
# intermediate files: PDDL files, BFWS output, plan.json / plan_sa.json.
TMP_DIR = "./tmp"

def clean_tmp():
    """Delete the scratch folder and everything in it."""
    shutil.rmtree(TMP_DIR, ignore_errors=True)

def check_bfws():
    """Fail early (before any LLM call) if the BFWS planner binary cannot be found."""
    bfws = os.environ.get("BFWS_BIN", "bfws")
    if shutil.which(bfws) is None:
        raise FileNotFoundError(f"BFWS planner '{bfws}' not found. Build LAPKT BFWS and set BFWS_BIN=/path/to/bfws.")

domain_definition = """
from unified_planning.shortcuts import *
from unified_planning.model.multi_agent import *
import sys, os
import json, time
from unified_planning.plot import plot_partial_order_plan
from unified_planning.plans import PlanKind
from unified_planning.io import PDDLWriter
import networkx as nx
import subprocess
## Define the domain
##############################################################################################
# UserTypes
area = UserType("area")
arm = UserType("arm")
obj_type = UserType("obj_type")
point_type = UserType("point_type")

# Universal Fluents

# point properties
is_releasable = Fluent("is_releasable", BoolType(), arg=point_type) # determine if it is allowed to release some obejct at this point
is_accessible = Fluent("is_accessible", BoolType(), arg=point_type) # determine if this point is accessible
# object properties
is_graspable = Fluent("is_graspable", BoolType(), arg=obj_type) # determine if the object is_graspable
is_free = Fluent("is_free", BoolType(), arg=obj_type) # determine if the object is not grasped or not. free means is ready to be grasped.
# arm properties
control = Fluent("control", BoolType(), arg=arm) # determine the control of the arm for different agents
available = Fluent("available", BoolType(), arg=arm) # determine the availability of the arm

arm_at = Fluent("arm_at", BoolType(), arg1=arm, arg2=area)  # determine the location of the arm
arm_access = Fluent("arm_access", BoolType(), arg1=arm, arg2=area) # determine the accessibility of the arm
lifting = Fluent("lifting", BoolType(), arg1=obj_type, arg2=arm) # determine the lifting status of the object
object_at_area = Fluent("object_at_area", BoolType(), arg1=obj_type, arg2=area) # determine which area that the object locates
object_at_point = Fluent("object_at_point", BoolType(), arg1=obj_type, arg2=point_type) # determine which point that the object locates. NOTE: It is not allowed to set one single object to two points at the same time!!
point_at = Fluent("point_at", BoolType(), arg1=point_type, arg2=area) # determine which point is at which area
water_in_cup = Fluent("water_in_cup", BoolType(), arg1=obj_type) # determine if the cup contains water
object_on_object = Fluent("object_on_object", BoolType(), arg1=obj_type, arg2=obj_type) # determine if the object (arg1) is over another object (arg2) (contacted). Useful when placing block on another block or placing objects on the bowl. NOTE: the point_type is not applicable here.
object_above_object = Fluent("above", BoolType(), arg1=obj_type, arg2=obj_type) # determine if the object (arg1) is above another object (arg2) (not contacted). Useful before executing the placing or pouring action. NOTE: the point_type is not applicable here.

# Universal Objects
left_area = Object("left_area", area)
right_area = Object("right_area", area)
overlap_area = Object("overlap_area", area)
left_arm = Object("left_arm", arm)
right_arm = Object("right_arm", arm)
areas = [left_area, right_area, overlap_area]

# task_related_objects
overlap_point_1 = Object("overlap_point_1", point_type)
overlap_point_2 = Object("overlap_point_2", point_type)
##############################################################################################
"""

ma_problem_initialization = """
## Define the problem and set up the initial values
##############################################################################################
problem = MultiAgentProblem("ma_bi_pddl")
problem.add_objects([left_area, right_area, overlap_area, left_arm, right_arm])
problem.ma_environment.add_fluent(arm_at, default_initial_value=False)
problem.ma_environment.add_fluent(arm_access, default_initial_value=False)
problem.ma_environment.add_fluent(available, default_initial_value=False)
problem.ma_environment.add_fluent(lifting, default_initial_value=False)
problem.ma_environment.add_fluent(water_in_cup, default_initial_value=False)
problem.ma_environment.add_fluent(object_at_area, default_initial_value=False)
problem.ma_environment.add_fluent(object_at_point, default_initial_value=False)
problem.ma_environment.add_fluent(object_on_object, default_initial_value=False)
problem.ma_environment.add_fluent(object_above_object, default_initial_value=False)
problem.ma_environment.add_fluent(is_graspable, default_initial_value=False)
problem.ma_environment.add_fluent(point_at, default_initial_value=False)
problem.ma_environment.add_fluent(is_releasable, default_initial_value=True)
problem.ma_environment.add_fluent(is_accessible, default_initial_value=True)
problem.ma_environment.add_fluent(is_free, default_initial_value=True)

problem.set_initial_value(arm_at(left_arm, left_area), True)
problem.set_initial_value(arm_at(left_arm, right_area), False)
problem.set_initial_value(arm_at(left_arm, overlap_area), False)
problem.set_initial_value(arm_access(left_arm, left_area), True)
problem.set_initial_value(arm_access(left_arm, overlap_area), True)
problem.set_initial_value(arm_access(left_arm, right_area), False)
problem.set_initial_value(available(left_arm), True)
problem.set_initial_value(arm_at(right_arm, right_area), True)
problem.set_initial_value(arm_at(right_arm, overlap_area), False)
problem.set_initial_value(arm_at(right_arm, left_area), False)
problem.set_initial_value(arm_access(right_arm, right_area), True)
problem.set_initial_value(arm_access(right_arm, overlap_area), True)
problem.set_initial_value(arm_access(right_arm, left_area), False)
problem.set_initial_value(available(right_arm), True)

problem.set_initial_value(point_at(overlap_point_1, overlap_area), True)
problem.set_initial_value(point_at(overlap_point_2, overlap_area), True)
problem.set_initial_value(is_releasable(overlap_point_1), True)
problem.set_initial_value(is_releasable(overlap_point_2), True)
problem.set_initial_value(is_accessible(overlap_point_1), True)
problem.set_initial_value(is_accessible(overlap_point_2), True)
"""

# Scratch folder setup inside the generated planning scripts
solver_common = """
TMP_DIR = "./tmp"
os.makedirs(TMP_DIR, exist_ok=True)
"""

# Runs the local BFWS planner on PDDL files and returns [(action_name, (param, ...)), ...]
bfws_solver_helper = """
import shutil
# Path to the LAPKT BFWS planner binary: set the BFWS_BIN environment variable, or put `bfws` on your PATH.
BFWS_BIN = os.environ.get("BFWS_BIN", "bfws")
if shutil.which(BFWS_BIN) is None:
    raise FileNotFoundError(f"BFWS planner '{BFWS_BIN}' not found. Build LAPKT BFWS and set BFWS_BIN=/path/to/bfws.")
SA_DOMAIN_FILE = os.path.join(TMP_DIR, "sa_domain.pddl")
SA_PROBLEM_FILE = os.path.join(TMP_DIR, "sa_problem.pddl")
def solve_with_bfws(domain_file, problem_file, output_file=os.path.join(TMP_DIR, "bfws_plan.txt")):
    domain_file, problem_file, output_file = map(os.path.abspath, (domain_file, problem_file, output_file))
    if os.path.exists(output_file):
        os.remove(output_file)
    command = [BFWS_BIN, "--domain", domain_file, "--problem", problem_file,
               "--output", output_file, "--max_novelty", "2", "--k-BFWS", "true"]
    # cwd=TMP_DIR so BFWS's own side file (execution.details) also lands in the tmp folder
    result = subprocess.run(command, capture_output=True, text=True, cwd=TMP_DIR)
    if not os.path.exists(output_file):
        raise RuntimeError("BFWS produced no plan file. stderr: " + result.stderr)
    plan = []
    with open(output_file, "r") as f:
        for line in f:
            line = line.strip().strip("()")
            if not line:
                continue
            parts = line.lower().split()
            plan.append((parts[0], tuple(parts[1:])))
    return plan
"""

# LLM+MAP solver, part 1: multi-agent FMAP planner (left and right arm as two agents)
fmap_solver_definition = """
# Define the agents and solve the problem
##############################################################################################
Left = Agent("left", problem)
Right = Agent("right", problem)
Left.add_actions(action_blocks)
Left.add_private_fluent(control, default_initial_value=False)
Right.add_actions(action_blocks)
Right.add_private_fluent(control, default_initial_value=False)
problem.add_agent(Left)
problem.add_agent(Right)
problem.set_initial_value(Dot(Left, control(left_arm)), True)
problem.set_initial_value(Dot(Right, control(right_arm)), True)

start_time = time.time()
up.shortcuts.get_environment().credits_stream = None
with OneshotPlanner(name='fmap') as planner:
    result = planner.solve(problem, timeout=60)
    if result.status == up.engines.PlanGenerationResultStatus.SOLVED_SATISFICING:
    # if result is not None:
        print(f"{planner.name} found this plan: {result.plan}")
        LEFT_PLAN_FMAP, RIGHT_PLAN_FMAP = plan2action(result.plan)
        print("Steps for Partial-order Plan: ", len(LEFT_PLAN_FMAP))
        fmap_plan_length = len(LEFT_PLAN_FMAP)
        fmap_found = True
    else:
        fmap_plan_length = 100
        fmap_found = False
        print("No FMAP plan found.")
print("The time spent for generating the plan: ", time.time() - start_time)
# plot_partial_order_plan(result.plan, figsize=(10, 10), node_size=4000, font_size=10, top_bottom=True)
"""

# LLM+MAP solver, part 2: single-agent BFWS planner -> partial-order schedule; keep the shorter of FMAP/BFWS
bfws_solver_definition = """

###### Actions definition for SA problem and solver
############################################################################################################
move_to.clear_effects()
x = move_to.parameter("x")
y = move_to.parameter("y")
z = move_to.parameter("z")
w = move_to.parameter("w")
area_variable = Variable("area", area)
obj_variable = Variable("objs", obj_type)
point_variable = Variable("points", point_type)

area_variable = Variable("area", area)
obj_variable = Variable("objs", obj_type)
point_variable = Variable("points", point_type)

for aa in areas:
    move_to.add_effect(object_at_area(x, aa), False, condition=Not(Equals(aa, w)))
    move_to.add_effect(arm_at(y, aa), False, condition=Not(Equals(aa, w)))
for pp in points:
    move_to.add_effect(is_accessible(pp), True, condition=And(object_at_point(x, pp), Not(Equals(pp, z))))
    move_to.add_effect(object_at_point(x, pp), False, condition=And(object_at_point(x, pp), Not(Equals(pp, z))))
for oo in objs:
    move_to.add_effect(object_above_object(x, oo), False, condition=Not(Equals(oo, x)))
    move_to.add_effect(object_above_object(oo, x), False, condition=Not(Equals(oo, x)))
    move_to.add_effect(object_on_object(oo, x), False, condition=Not(Equals(oo, x)))
    move_to.add_effect(object_on_object(x, oo), False, condition=Not(Equals(oo, x)))
move_to.add_effect(object_at_area(x, w), True)
move_to.add_effect(arm_at(y, w), True)
move_to.add_effect(object_at_point(x, z), True)
move_to.add_effect(is_accessible(z), False)

co_hold = InstantaneousAction("co_hold", x=obj_type, y=arm, z=arm)
x = co_hold.parameter("x")
y = co_hold.parameter("y")
z = co_hold.parameter("z")

co_hold.add_precondition(available(y))
co_hold.add_precondition(available(z))
co_hold.add_precondition(object_at_area(x, overlap_area))
co_hold.add_precondition(Not(is_graspable(x)))
co_hold.add_precondition(Not(Equals(y, z)))

co_hold.add_effect(available(left_arm), False)
co_hold.add_effect(available(right_arm), False)
co_hold.add_effect(lifting(x, left_arm), True)
co_hold.add_effect(lifting(x, right_arm), True)

co_move_to = InstantaneousAction("co_move_to", x=obj_type, y=point_type, z=arm, w=arm)
x = co_move_to.parameter("x")
y = co_move_to.parameter("y")
z = co_move_to.parameter("z")
w = co_move_to.parameter("w")

co_move_to.add_precondition(object_at_area(x, overlap_area))
co_move_to.add_precondition(point_at(y, overlap_area))
co_move_to.add_precondition(lifting(x, z))
co_move_to.add_precondition(lifting(x, z))
co_move_to.add_precondition(Not(Equals(z, w)))

co_move_to.add_effect(object_at_point(x, y), True)

####  Single-agent Problem Definition
sa_problem = Problem("seq_bi_manipulation")
for fluent in problem.ma_environment.fluents:
    sa_problem.add_fluent(fluent)
for object in problem._objects:
    sa_problem.add_object(object)
sa_problem.add_fluent(control)
for action in action_blocks:
    if action.name == "co_hold":
        sa_problem.add_action(co_hold)
    elif action.name == "co_move_to":
        sa_problem.add_action(co_move_to)
    else:
        sa_problem.add_action(action)
for fluent, value in problem.initial_values.items():
    try:
        sa_problem.set_initial_value(fluent, value)
    except Exception as e:
        pass
sa_problem.set_initial_value(control(left_arm), True)
sa_problem.set_initial_value(control(right_arm), True)
for goal in problem.goals:
    sa_problem.add_goal(goal)

writer = PDDLWriter(sa_problem)
writer.write_domain(SA_DOMAIN_FILE)
writer.write_problem(SA_PROBLEM_FILE)

from unified_planning.plans import PlanKind
from unified_planning.plans import SequentialPlan, ActionInstance
from unified_planning.model import Problem, InstantaneousAction, Object
start_time = time.time()
bfws_plan_steps = solve_with_bfws(SA_DOMAIN_FILE, SA_PROBLEM_FILE)
action_instances = []
for action_name, params_names in bfws_plan_steps:
    action = sa_problem.action(action_name)
    param_objects = tuple(sa_problem.object(name) for name in params_names)
    action_instances.append(ActionInstance(action, param_objects))
bfws_plan = SequentialPlan(action_instances)
partial_order_plan = bfws_plan.convert_to(PlanKind.PARTIAL_ORDER_PLAN, problem)
print("The POP plan from BFWS planner is")
print(partial_order_plan)

with PlanValidator(name="sequential_plan_validator") as validator:
    validate_result = validator.validate(sa_problem, bfws_plan)
    print(validate_result)

    def schedule_actions(G):
        sorted_actions = list(nx.topological_sort(G))
        schedule = defaultdict(lambda: {"left_arm": "wait", "right_arm": "wait"})
        
        arm_tasks = {"left_arm": [], "right_arm": []}
        for action in sorted_actions:
            if "left_arm" in str(action):
                arm_tasks["left_arm"].append(action)
            elif "right_arm" in str(action):
                arm_tasks["right_arm"].append(action)

        time_step = 0
        while sorted_actions:
            executed = set()
            for arm in ["left_arm", "right_arm"]:
                for action in sorted_actions:
                    if action in arm_tasks[arm] and G.in_degree(action) == 0:
                        schedule[time_step][arm] = action
                        executed.add(action)
                        break
            
            if not executed:
                time_step += 1
                continue

            sorted_actions = [a for a in sorted_actions if a not in executed]
            G.remove_nodes_from(executed)
            time_step += 1
        
        return schedule

if 'INVALID' in str(validate_result.status):
    bfws_plan_length = 100
    print("No BFWS plan found!!")
else:
    schedule = schedule_actions(partial_order_plan._graph)
    LEFT_PLAN_BFWS, RIGHT_PLAN_BFWS = [], []
    for i, action_list in schedule.items():
        LEFT_PLAN_BFWS.append(str(action_list['left_arm']))
        RIGHT_PLAN_BFWS.append(str(action_list['right_arm']))
    plans = {"Left": LEFT_PLAN_BFWS, "Right": LEFT_PLAN_BFWS}
    bfws_plan_length = len(LEFT_PLAN_BFWS)
    print("Steps for Partial-order Plan: ", len(LEFT_PLAN_BFWS))
    print("The time spent for generating the BFWS plan: ", time.time() - start_time)

    if fmap_plan_length <= bfws_plan_length:
        plans = {"Left": LEFT_PLAN_FMAP, "Right": RIGHT_PLAN_FMAP}
        print("The Partial-order plan from FMAP is the optimal!")
        with open(os.path.join(TMP_DIR, "plan.json"), "w") as f:
            json.dump(plans, f)
    else:
        plans = {"Left": LEFT_PLAN_BFWS, "Right": RIGHT_PLAN_BFWS}
        print("The Partial-order plan from BFWS is the optimal!")
        with open(os.path.join(TMP_DIR, "plan.json"), "w") as f:
            json.dump(plans, f)
"""

# FMAP only (BFWS disabled): save the FMAP plan directly
fmap_only_output = """
if fmap_found:
    print("The Partial-order plan from FMAP is selected (BFWS disabled).")
    with open(os.path.join(TMP_DIR, "plan.json"), "w") as f:
        json.dump({"Left": LEFT_PLAN_FMAP, "Right": RIGHT_PLAN_FMAP}, f)
"""

map_solver_definition = solver_common + bfws_solver_helper + fmap_solver_definition + bfws_solver_definition
fmap_only_solver_definition = solver_common + fmap_solver_definition + fmap_only_output

def get_map_solver_definition(use_bfws=True):
    """Solver code for LLM+MAP: FMAP + BFWS (default), or FMAP only."""
    return map_solver_definition if use_bfws else fmap_only_solver_definition

sa_solver_definition = solver_common + bfws_solver_helper + """
###### Actions definition for SA problem and solver
############################################################################################################
move_to.clear_effects()
x = move_to.parameter("x")
y = move_to.parameter("y")
z = move_to.parameter("z")
w = move_to.parameter("w")
area_variable = Variable("area", area)
obj_variable = Variable("objs", obj_type)
point_variable = Variable("points", point_type)

area_variable = Variable("area", area)
obj_variable = Variable("objs", obj_type)
point_variable = Variable("points", point_type)

for aa in areas:
    move_to.add_effect(object_at_area(x, aa), False, condition=Not(Equals(aa, w)))
    move_to.add_effect(arm_at(y, aa), False, condition=Not(Equals(aa, w)))
for pp in points:
    move_to.add_effect(is_accessible(pp), True, condition=And(object_at_point(x, pp), Not(Equals(pp, z))))
    move_to.add_effect(object_at_point(x, pp), False, condition=And(object_at_point(x, pp), Not(Equals(pp, z))))
for oo in objs:
    move_to.add_effect(object_above_object(x, oo), False, condition=Not(Equals(oo, x)))
    move_to.add_effect(object_above_object(oo, x), False, condition=Not(Equals(oo, x)))
    move_to.add_effect(object_on_object(oo, x), False, condition=Not(Equals(oo, x)))
    move_to.add_effect(object_on_object(x, oo), False, condition=Not(Equals(oo, x)))
move_to.add_effect(object_at_area(x, w), True)
move_to.add_effect(arm_at(y, w), True)
move_to.add_effect(object_at_point(x, z), True)
move_to.add_effect(is_accessible(z), False)

co_hold = InstantaneousAction("co_hold", x=obj_type, y=arm, z=arm)
x = co_hold.parameter("x")
y = co_hold.parameter("y")
z = co_hold.parameter("z")

co_hold.add_precondition(available(y))
co_hold.add_precondition(available(z))
co_hold.add_precondition(object_at_area(x, overlap_area))
co_hold.add_precondition(Not(is_graspable(x)))
co_hold.add_precondition(Not(Equals(y, z)))

co_hold.add_effect(available(left_arm), False)
co_hold.add_effect(available(right_arm), False)
co_hold.add_effect(lifting(x, left_arm), True)
co_hold.add_effect(lifting(x, right_arm), True)

co_move_to = InstantaneousAction("co_move_to", x=obj_type, y=point_type, z=arm, w=arm)
x = co_move_to.parameter("x")
y = co_move_to.parameter("y")
z = co_move_to.parameter("z")
w = co_move_to.parameter("w")

co_move_to.add_precondition(object_at_area(x, overlap_area))
co_move_to.add_precondition(point_at(y, overlap_area))
co_move_to.add_precondition(lifting(x, z))
co_move_to.add_precondition(lifting(x, z))
co_move_to.add_precondition(Not(Equals(z, w)))

co_move_to.add_effect(object_at_point(x, y), True)

####  Single-agent Problem Definition
sa_problem = Problem("seq_bi_manipulation")
for fluent in problem.ma_environment.fluents:
    sa_problem.add_fluent(fluent)
for object in problem._objects:
    sa_problem.add_object(object)
sa_problem.add_fluent(control)
for action in action_blocks:
    if action.name == "co_hold":
        sa_problem.add_action(co_hold)
    elif action.name == "co_move_to":
        sa_problem.add_action(co_move_to)
    else:
        sa_problem.add_action(action)

for fluent, value in problem.initial_values.items():
    try:
        sa_problem.set_initial_value(fluent, value)
    except Exception as e:
        pass
sa_problem.set_initial_value(control(left_arm), True)
sa_problem.set_initial_value(control(right_arm), True)
for goal in problem.goals:
    sa_problem.add_goal(goal)

writer = PDDLWriter(sa_problem)
writer.write_domain(SA_DOMAIN_FILE)
writer.write_problem(SA_PROBLEM_FILE)

from unified_planning.plans import PlanKind
from unified_planning.plans import SequentialPlan, ActionInstance
from unified_planning.model import Problem, InstantaneousAction, Object
start_time = time.time()
try:
    bfws_plan_steps = solve_with_bfws(SA_DOMAIN_FILE, SA_PROBLEM_FILE)
    action_instances = []
    for action_name, params_names in bfws_plan_steps:
        action = sa_problem.action(action_name)
        param_objects = tuple(sa_problem.object(name) for name in params_names)
        action_instances.append(ActionInstance(action, param_objects))
    bfws_plan = SequentialPlan(action_instances)
    print("Sequential plan: ", bfws_plan)
    with open(os.path.join(TMP_DIR, "plan_sa.json"), "w") as f:
        json.dump(bfws_plan_steps, f, indent=4)
    print("plan step length", len(bfws_plan_steps))
    print("The time spent for generating the plan: ", time.time() - start_time)
except Exception as e:
    print("No plan found! Error:", e)
"""

# sa_solver_definition = """
# sa_problem = Problem("seq_bi_pddl")

# for fluent in problem.ma_environment.fluents:
#     sa_problem.add_fluent(fluent)
# for object in problem._objects:
#     sa_problem.add_object(object)
# sa_problem.add_fluent(control)
# sa_problem.add_actions(action_blocks)

# for fluent, value in problem.initial_values.items():
#     try:
#         sa_problem.set_initial_value(fluent, value)
#     except Exception as e:
#         pass
# sa_problem.set_initial_value(control(left_arm), True)
# sa_problem.set_initial_value(control(right_arm), True)

# for goal in problem.goals:
#     sa_problem.add_goal(goal)

# # writer = PDDLWriter(sa_problem)
# # writer.write_domain('sa_stackblocks_domain.pddl')
# # writer.write_problem('sa_stackblocks_problem.pddl')

# start_time = time.time()
# with OneshotPlanner(name='fast-downward-opt') as planner:
#     result = planner.solve(sa_problem)
#     if result.status == up.engines.PlanGenerationResultStatus.SOLVED_SATISFICING:
#     # if result is not None:
#         print(f"{planner.name} found this plan: {result.plan}")
#         # LEFT_PLAN, RIGHT_PLAN = plan2action(result.plan)
#         # plans = {"Left": LEFT_PLAN, "Right": RIGHT_PLAN}
#         # with open(os.path.join(TMP_DIR, "plan.json"), "w") as f:
#         #     json.dump(plans, f)
#     else:
#         print("No plan found.")

# # with PlanValidator() as validator:
# #     validate_result = validator.validate(sa_problem, result.plan)
# #     print(validate_result)

# partial_order_plan = result.plan.convert_to(PlanKind.PARTIAL_ORDER_PLAN, sa_problem)
# print(partial_order_plan)
# """

plan2action = """
from collections import defaultdict, deque

def plan2action(plan):
    # Get the adjacency list from the plan
    adjacency_list = plan.get_adjacency_list

    # Create a list of actions and assign indices
    actions_list = list(adjacency_list.keys())
    action_indices = {action: idx for idx, action in enumerate(actions_list)}

    # Build the actions dictionary
    actions = {}
    for idx, action in enumerate(actions_list):
        action_str = str(action)
        # Split the action string into agent and action details
        parts = action_str.split('.', 1)
        if len(parts) == 2:
            agent, action_detail = parts
        else:
            agent = ''
            action_detail = action_str
        actions[idx] = (agent, action_detail)

    # Build the constraints list
    constraints = []
    for before_action, after_actions in adjacency_list.items():
        before_idx = action_indices[before_action]
        for after_action in after_actions:
            after_idx = action_indices[after_action]
            constraints.append((before_idx, after_idx))

    graph = defaultdict(list)
    in_degree = {k: 0 for k in actions.keys()}

    for before, after in constraints:
        graph[before].append(after)
        in_degree[after] += 1

    # Perform level-order topological sort
    queue = deque()
    # Initialize the queue with actions that have in-degree of 0
    for action in actions:
        if in_degree[action] == 0:
            queue.append(action)

    time_steps = []  # List of lists of actions at each time step

    while queue:
        # Actions that can be executed in this time step
        level_actions = []
        next_queue = deque()
        # Collect all actions with in-degree 0
        while queue:
            action = queue.popleft()
            level_actions.append(action)
        # Add actions to the time steps
        time_steps.append(level_actions)
        # Decrease in-degree of successors
        for action in level_actions:
            for successor in graph[action]:
                in_degree[successor] -= 1
                if in_degree[successor] == 0:
                    next_queue.append(successor)
        queue = next_queue

    # Build per-agent plans
    agents = ['Left', 'Right']
    agent_plans = {agent: [] for agent in agents}

    # Find the maximum number of time steps
    max_steps = len(time_steps)

    # for t in range(max_steps):
    #     # Actions executed at time step t
    #     current_actions = time_steps[t]
    #     # Map agent to their action at this time step
    #     agent_action = {agent: 'wait' for agent in agents}
    #     for action_num in current_actions:
    #         agent, action_desc = actions[action_num]
    #         agent_action[agent] = actions[action_num][1]
    #         if "co_" in actions[action_num][1]:
    #             for agent in agents:
    #                 agent_action[agent] = actions[action_num][1]
    #     # Append actions or 'wait' to each agent's plan
    #     for agent in agents:
    #         agent_plans[agent].append(agent_action[agent])
    # return agent_plans['Left'], agent_plans['Right']
    for t in range(max_steps):
        # Actions executed at time step t
        current_actions = time_steps[t]
        # Map agent to their action at this time step
        agent_action = {agent: 'wait' for agent in agents}
        for action_num in current_actions:
            agent, action_desc = actions[action_num]
            if 'left_arm' in actions[action_num][1]:
                agent_action['Left'] = actions[action_num][1]
            else:
                agent_action['Right'] = actions[action_num][1]
            # agent_action[agent] = actions[action_num][1]
            if "co_" in actions[action_num][1]:
                for agent in agents:
                    agent_action[agent] = actions[action_num][1]
        # Append actions or 'wait' to each agent's plan
        for agent in agents:
            agent_plans[agent].append(agent_action[agent])
    return agent_plans['Left'], agent_plans['Right']
"""

# Please make sure that all the relative predicates are defined. For example, you should define is_different(obj1, obj2) and is_different(obj2, obj1) as both True.


# There are spatio-temporal control rules for two hands: 
# 1) Uncoordinated Control: two hands act and manipulate independently in their own working spaces (i.e., left area and right area respectively), referred to as unimanual manipulation for each hand. For safety concerns and better efficiency, the left area is not accessible by the right hand, and vice versa.
# 2) Coordinated Control: two hands act and manipulate dependently in terms of either spatial or temporal relation, e.g., co-working on the overlap area (with points: overlap_point_1, overlap_point_2). The collaboration between two hands can be either asynchronous or synchronous — An asynchronous type of control involves one hand constructing pre-conditions for the other, while a synchronous control indicates a, usually precise, mutual dependency between them.


### action definitions: 
grasp_act = """ \n
grasp = InstantaneousAction("grasp", x=obj_type, y=arm, z=area)
x = grasp.parameter("x")
y = grasp.parameter("y")
z = grasp.parameter("z")

grasp.add_precondition(control(y))
grasp.add_precondition(arm_access(y, z))
grasp.add_precondition(object_at_area(x, z))
grasp.add_precondition(available(y))
grasp.add_precondition(is_graspable(x))
grasp.add_precondition(is_free(x))

grasp.add_effect(lifting(x, y), True)
grasp.add_effect(available(y), False)
grasp.add_effect(is_free(x), False)
"""

move_to_act = """\n
move_to = InstantaneousAction("move_to", x=obj_type, y=arm, z=point_type, w=area)
x = move_to.parameter("x")
y = move_to.parameter("y")
z = move_to.parameter("z")
w = move_to.parameter("w")

move_to.add_precondition(control(y))
move_to.add_precondition(is_graspable(x))
move_to.add_precondition(lifting(x, y))
move_to.add_precondition(arm_access(y, w))
move_to.add_precondition(point_at(z, w))
move_to.add_precondition(is_accessible(z))

for aa in areas: move_to.add_effect(object_at_area(x, aa), False)
for aa in areas: move_to.add_effect(arm_at(y, aa), False)
# for pp in points: 
#     if (pp != z) and object_at_point(x, pp):
#         move_to.add_effect(is_accessible(pp), True)
for pp in points: move_to.add_effect(object_at_point(x, pp), False)
for p in objs:
    if p != x: 
        move_to.add_effect(object_above_object(x, p), False)
        move_to.add_effect(object_above_object(p, x), False)
    if object_on_object(x, p):
        move_to.add_effect(object_on_object(x, p), False)
move_to.add_effect(object_at_area(x, w), True)
move_to.add_effect(arm_at(y, w), True)
move_to.add_effect(object_at_point(x, z), True)
move_to.add_effect(is_accessible(z), False)
"""

move_above_act = """\n
move_above = InstantaneousAction("move_above", x=obj_type, y=arm, z=obj_type, w=area)
x = move_above.parameter("x")
y = move_above.parameter("y")
z = move_above.parameter("z")
w = move_above.parameter("w")

move_above.add_precondition(control(y))
move_above.add_precondition(arm_access(y, w))
move_above.add_precondition(object_at_area(z, w))
move_above.add_precondition(lifting(x, y))
move_above.add_precondition(x!=z)
move_above.add_precondition(is_graspable(x))

for aa in areas: move_above.add_effect(object_at_area(x, aa), False)
for aa in areas: move_above.add_effect(arm_at(y, aa), False)
for pp in points: move_above.add_effect(object_at_point(x, pp), False)
for p in objs:
    if p != x: 
        move_above.add_effect(object_above_object(x, p), False)
        move_above.add_effect(object_above_object(p, x), False)
        move_above.add_effect(object_on_object(x, p), False)
move_above.add_effect(object_above_object(x, z), True)
move_above.add_effect(object_at_area(x, w), True)
move_above.add_effect(arm_at(y, w), True)
"""

pour_act = """\n
pour = InstantaneousAction("pour", x=obj_type, y=arm, z=obj_type)
x = pour.parameter("x")
y = pour.parameter("y")
z = pour.parameter("z")
pour.add_precondition(lifting(x, y))
pour.add_precondition(control(y))
pour.add_precondition(water_in_cup(x))
pour.add_precondition(object_above_object(x, z))
pour.add_effect(water_in_cup(z), True)
pour.add_effect(water_in_cup(x), False)
"""

push_act = """\n
push = InstantaneousAction("push", x=obj_type, y=arm, z=area, w=area)
x = push.parameter("x")
y = push.parameter("y")
z = push.parameter("z")
w = push.parameter("w")

push.add_precondition(object_at_area(x, z))
push.add_precondition(control(y))
push.add_precondition(arm_access(y, w))
push.add_precondition(arm_access(y, z))
push.add_precondition(available(y))
push.add_precondition(Not(is_graspable(x)))

push.add_effect(object_at_area(x, w), True)
push.add_effect(object_at_area(x, z), False)
"""

release_act = """\n
release = InstantaneousAction("release", x=obj_type, y=arm, z=point_type)
x = release.parameter("x")
y = release.parameter("y")
z = release.parameter("z")

release.add_precondition(control(y))
release.add_precondition(lifting(x, y))
release.add_precondition(object_at_point(x, z))
release.add_precondition(is_releasable(z))
release.add_precondition(is_graspable(x))

release.add_effect(available(y), True)
release.add_effect(lifting(x, y), False)
release.add_effect(is_free(x), True)
release.add_effect(is_accessible(z), False)
"""

place_block_act = """\n
place = InstantaneousAction("place", x=obj_type, y=arm, z=obj_type, w=area)
x = place.parameter("x")
y = place.parameter("y")
z = place.parameter("z")
w = place.parameter("w")

place.add_precondition(control(y))
place.add_precondition(lifting(x, y))
# if y == left_arm: place.add_precondition(Not(lifting(x, right_arm)))
# if y == right_arm: place.add_precondition(Not(lifting(x, left_arm)))
place.add_precondition(arm_at(y, w))
place.add_precondition(object_at_area(x, w))
place.add_precondition(object_at_area(z, w))
place.add_precondition(object_above_object(x, z))
place.add_precondition(is_free(z))

place.add_effect(available(y), True)
place.add_effect(lifting(x, y), False)
place.add_effect(is_free(x), True)
place.add_effect(object_on_object(x, z), True)
place.add_effect(is_free(z), False)
"""

place_fruit_act = """\n
place = InstantaneousAction("place", x=obj_type, y=arm, z=obj_type, w=area)
x = place.parameter("x")
y = place.parameter("y")
z = place.parameter("z")
w = place.parameter("w")

place.add_precondition(control(y))
place.add_precondition(lifting(x, y))
# if y == left_arm: place.add_precondition(Not(lifting(x, right_arm)))
# if y == right_arm: place.add_precondition(Not(lifting(x, left_arm)))
place.add_precondition(arm_at(y, w))
place.add_precondition(object_at_area(x, w))
place.add_precondition(object_at_area(z, w))
place.add_precondition(object_above_object(x, z))
place.add_precondition(is_free(z))

place.add_effect(available(y), True)
place.add_effect(lifting(x, y), False)
place.add_effect(is_free(x), True)
place.add_effect(object_on_object(x, z), True)
"""

co_hold_act = """\n
co_hold = InstantaneousAction("co_hold", x=obj_type)
x = co_hold.parameter("x")

co_hold.add_precondition(available(left_arm))
co_hold.add_precondition(available(right_arm))
co_hold.add_precondition(object_at_area(x, overlap_area))
co_hold.add_precondition(Not(is_graspable(x)))
co_hold.add_effect(available(left_arm), False)
co_hold.add_effect(available(right_arm), False)
co_hold.add_effect(lifting(x, left_arm), True)
co_hold.add_effect(lifting(x, right_arm), True)
"""

co_move_to_act = """\n
co_move_to = InstantaneousAction("co_move_to", x=obj_type, y=point_type)
x = co_move_to.parameter("x")
y = co_move_to.parameter("y")

co_move_to.add_precondition(object_at_area(x, overlap_area))
co_move_to.add_precondition(point_at(y, overlap_area))
co_move_to.add_precondition(lifting(x, left_arm))
co_move_to.add_precondition(lifting(x, right_arm))

co_move_to.add_effect(object_at_point(x, y), True)
"""

co_move_above_act = """\n
co_move_above = InstantaneousAction("co_move_above", x=obj_type, y=obj_type)
x = co_move_above.parameter("x")
y = co_move_above.parameter("y")

co_move_above.add_precondition(object_at_area(x, overlap_area))
co_move_above.add_precondition(object_at_area(y, overlap_area))
co_move_above.add_precondition(lifting(x, left_arm))
co_move_above.add_precondition(lifting(x, right_arm))

co_move_above.add_effect(object_above_object(x, y), True)
"""

co_release_act = """\n
co_release = InstantaneousAction("co_release", x=obj_type)
x = co_release.parameter("x")

co_release.add_precondition(object_at_area(x, overlap_area))
co_release.add_precondition(lifting(x, left_arm))
co_release.add_precondition(lifting(x, right_arm))
co_release.add_precondition(Not(is_graspable(x)))

co_release.add_effect(lifting(x, left_arm), False)
co_release.add_effect(lifting(x, right_arm), False)
co_release.add_effect(available(left_arm), True)
co_release.add_effect(available(right_arm), True)
"""

def choose_actions(task_name):
    if task_name == "StackBlock":
        action_chunk = f"""
{grasp_act} \n {move_to_act} \n {move_above_act} \n {place_block_act} \n {release_act} \n
action_blocks = [grasp, move_to, move_above, place, release]
"""
    elif task_name == "ServeWater":
        action_chunk = f"""
{grasp_act} \n {move_to_act} \n {move_above_act} \n {pour_act} \n {release_act} \n
action_blocks = [grasp, move_to, move_above, pour, release]
"""
    elif task_name == "ServeFruit":
        action_chunk = f"""
{grasp_act} \n {move_to_act} \n {move_above_act} \n {place_fruit_act} \n {push_act} \n {release_act} \n {co_hold_act} \n {co_move_to_act} \n
action_blocks = [grasp, move_to, move_above, place, push, release, co_hold, co_move_to]
"""
    else:
        print("You choose the wrong task name!")
        return None
    return action_chunk