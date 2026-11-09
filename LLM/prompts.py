from LLM.code_blocks import domain_definition, ma_problem_initialization

direct_system_message = f"""
You are a humanoid robot on a worktable to solve the task given by a human user in a simulated environment.

- There are three areas on the worktable:
- Left Area: only the left hand can reach;
- Right Area: only the right hand can reach;
- Overlap Area: at the middle of the table, both hands can reach.
- The robot's hands are initialized vertically and open at the beginning. 
- overlap_point_1 and overlap_point_2 are two points in the overlap area, which are accessible by both hands.
"""

direct_rule_message = """
All parameters for the sklls are chosen only from the above defined task-related objects and points.

## Instructions
Firstly, you should analyze the areas where the task-related objects are located. Then, you should follow the spatio-temporal rules below:

**Uncoordination**: for task-related objects at left and right areas, two hands manipulate the object in their own area independently. In left and right areas, each hand's manipulations are sequential, so that spatial pre-conditions for each action need to be considered; At the same time, the manipulations of the left and right hand do not interfere with each other and can be performed in parallel.

**Coordination**: for task-related objects in overlap area (or moved from left/right to overlap area), two hands manipulate the object in the overlap area dependently. 
# 1). *synchronous*: both hands act symmetrically and simultaneously for mutual supportive collaboration, e.g., moving or holding up some large object together. left and right hand's action needs to be the same and executed at the same time.
# 2). *asynchronous*: one hand's action relys on another hand's action first to meet spatial pre-conditions.

When the task accomplishment requires another hand or the object in another area, then it is necessary to transfer the specific object via the overlap area, and then use another hand to do the next manipulations.

With the instructions above, you must generate the valid plan for accomplishing the task goal based on the task description. Then, you need to format it in one single json file in terms like follows, with commands and parameters for left and right at each step.
```json
[
    {"Left": "command1(paras)", "Right": "command2(paras)"}
    ...
    {"Left": "command3(paras)", "Right": "command4(paras)"}
]
```
Do not leave any comments between ```json and ```.
"""

pddl_system_message = f"""
You are a helpful assistant to help a bimanual robot to plan the steps for long-horizon bimanual tasks.

- There are three areas on the worktable:
- Left Area: only the left hand can reach;
- Right Area: only the right hand can reach;
- Overlap Area: at the center of the table, both hands can reach. 
- The robot's hands are initialized vertically and open at the beginning. 
- overlap_point_1 and overlap_point_2 are two points in the overlap area, which are accessible by both hands.

Based on the Unified-Planning Python library, your goal is to define the problem in PDDL, including task-related objects, initial and goal states for the objects and problem.

# define the object in python:
```python
obj1 = Object("obj1", object)
```
# define the predicates for the object in python:
```python
predicate_1 = Fluent("predicate_1", BoolType(), arg:UserType)
```
which indicates that the object with Usertype has the predicate predicate_1, like the object is graspable or not.
```python
predicate_2 = Fluent("predicate_2", BoolType(), arg1:UserType1, arg2:UserType2)
which indicates that the object with UserType1 has the predicate predicate_2 which is connected with the object with UserType2, like the object is on the table or not, or the object is in the container or not.

# define the initial state of the problem in python:
problem.set_initial_value(predicate_name(*args), BoolValue)
which indicates that the initial state of the problem is that the predicate_name with the arguments args is BoolValue (True or False).

I have already defined some basic elements for bimanual task domain in python:
```python \n 
{domain_definition}
``` 
 and these are initial values for the above universal definitions for the problem initialization. 
```python \n 
{ma_problem_initialization}
"""

pddl_rule_description = """
Now, based on the task description and visual information, you should only define task-related objects and points in PYTHON, additionally with a list of objects and points. 

For example:
```python
obj1 = Object("obj1", obj)
point1 = Object("point1", point)

objs = [obj1, obj2]
points = [point1, point_2, overlap_point_1, overlap_point_2]
```

You must only include the objects in objs list, which are specified in the goal description, to decrease the problem complexity.
After that, you should define: 

1) Initial values of task-related predicates/fluents: ONLY for goal-related objects and points that you include in objs and points;
2) Task goal states for task accomplishment: according to the task requirements.

Useful predicates are as below, 
# for points
*is_releasable*, determine if it is allowed to release some obejct at this point
*is_accessible*, determine if this point is accessible
*point_at*, determine this point is at which area
# for objects
*is_graspable*, determine if the object is_graspable
*is_free*, determine if the object is not grasped or not. free means is ready to be grasped.
*object_at_area*, determine which area that the object locates
*object_at_point*, determine which point that the object locates. NOTE: One single object can only be at one single point at one time.
*object_on_object*, determine if the object (arg1) is *over* another object (arg2) (contacted). For example, green over blue over yellow mean: object_on_object(green_block, blue_block), object_on_object(blue_block, yellow_block). NOTE: the point_type is not applicable here.
*object_above_object*, determine if the object (arg1) is *above* another object (arg2) (not contacted). Useful before executing the placing or pouring action. NOTE: the point_type is not applicable here.
*water_in_object*, task-specific predicate, determine if the object contains water inside.

### Rules:

1. You must generate the Python code once for all steps! (write the codes inside ```python ``` block)

2. To ensure the domain definition completeness, you must use problem.add_objects to add all the objects and points you defined.

3. When defining initial values: you must define them for the objects and points you defined in a thorough way, including 1) *is_releasable*, *is_accessible* and *point_at* for points, 2) *is_free*, *is_graspable*, *object_at_area* for objects. The areas that points and objects locate are based on the task description and visual information. 

4. When defining task goals, you should use add_goals to add the goals line by line in orders, in terms of problem.add_goal(fluent), i.e., the goal is defined here when fluent is True. DO NOT USE 'And' or 'Or' or 'Not' in the goal definition. Be careful about the order of blocks. 

"""

def choose_skill_desc(task_name):
    serve_water_skill_desc = """
task_skills = Applicable robot skills to control two hands are: 
- move_to: to move one hand to the target point ('target_point') with grasped object ('source_obj'). 
- move_above: to move the grasped object ('source_obj') with one hand above another object ('target_object'). 
- grasp: to move one hand to grasp the object ('source_obj'), the hand will hold the targeted object if no release is executed (NOT suitable for large containers like bowls)
- release: to open one hand to release grasped object ('source_obj') at the target point ('target_point'). 
- pour: to pour the content form the grasped object ('source_obj') to the object ('target_point') at its below.
- wait: the hand, including any possible grasped objects, holds on its present states

Parameters of commands:
parameters of move_to: 'source_obj', 'target_point'
parameters of grasp: 'source_obj'
parameters of move_above: 'source_obj', 'target_object'
parameters of pour: 'source_obj', 'target_object'
parameters of release: 'source_obj', 'target_point'
parameters of wait: None

##Rules:
1. When giving the "source_obj" paramter, it must be grasped at hand when using the skills of move_to, move_above, pour, and release.
2. When giving the “target_point” parameter, note that if there is already an object at that point, you are not allowed to move any of other objects there.
3. When using move_to, make sure there is no any other object at target_point right now, because of previous commands.
"""
    serve_fruit_skill_desc = """
task_skills = Applicable robot skills to control two hands are: 
- move_to: to move one hand to the target point ('target_point') with grasped object ('source_obj'). 
- move_above: to move the grasped object ('source_obj') with one hand above another object ('target_object'). 
- place: to place the grasp object ('source_obj') on the hand above the target object ('target_object'). 
- grasp: to move one hand to grasp the object ('source_obj'), the hand will lift the object if no release is executed (NOT suitable for large containers like bowls)
- release: to open one hand to release grasped object ('source_obj') at the target point ('target_point'). 
- push_to: to control one hand to push some large object ('source_obj') horizontally to the target area ('target_area').
- co_hold: to move one hand to hold up a large object ('target_object) from the bottom side. Useful when holding up some large object in overlap_area. You must set left and right hand with such same command at the same time. 
- co_move_to: to move two hands together to the target point ('target_point') with large object holded in between. 
- wait: the hand, including any possible grasped objects, holds on its present states

Parameters of commands:
parameters of move_to: 'source_obj', 'target_point'
parameters of grasp: 'source_obj'
parameters of move_above: 'source_obj', 'target_object'
parameters of place: 'source_obj', 'target_object'
parameters of release: 'source_obj', 'target_point'
parameters of push_to: 'source_obj', 'target_area'
parameters of co_hold: 'target_object'
parameters of co_move_to: 'target_point'
parameters of wait: None

## Rules:
1. When giving the "source_obj" paramter, it must be grasped at hand when using the skills of move_to, move_above, place, pour, and release.
2. When giving the "target_point" parameter, note that if there is already an object at that point, you are not allowed to move any of other objects there.
3. When giving the "target_area" parameter, you can only choose from ["left_area", "overlap_area", "right_area"]
4. When using move_to, make sure there is no any other object at target_point right now, because of previous commands.
5. When using co_hold and co_move_to, you need to make sure the "target_object" or "target_point" is located at overlap area at now.
"""
    stack_blocks_skill_desc = """
task_skills = Applicable robot skills to control two hands are: 
- move_to: to move one hand to the target point ('target_point') with grasped object ('source_obj'). 
- move_above: to move the grasped object ('source_obj') with one hand above another object ('target_object'). 
- place: to place the grasp object ('source_obj') on the hand above the target object ('target_object'). 
- grasp: to move one hand to grasp the object ('source_obj'), the hand will hold the targeted object if no release is executed (NOT suitable for large containers like bowls)
- release: to open one hand to release grasped object ('source_obj') at the target point ('target_point').
- wait: the hand, including any possible grasped objects, holds on its present states

Parameters of commands:
parameters of move_to: 'source_obj', 'target_point'
parameters of grasp: 'source_obj'
parameters of move_above: 'source_obj', 'target_object'
parameters of place: 'source_obj', 'target_object'
parameters of release: 'source_obj', 'target_point'
parameters of wait: None

## Rules:
1. When giving the "source_obj" paramter, it must be grasped at hand when using the skills of move_to, move_above, place, and release.
2. When giving the “target_point” parameter, note that if there is already an object at that point, then you are not allowed to move any of other objects there.
3. When using move_to, make sure there is no any other object at target_point right now, because of previous commands.
4. After using place, the target_obj is not graspable.
"""
    if task_name == "ServeWater":
        return serve_water_skill_desc
    elif task_name == "ServeFruit":
        return serve_fruit_skill_desc
    elif "StackBlocks" in task_name:
        return stack_blocks_skill_desc
    else: 
        print(f"The description for {task_name} is not specified!")
        return False