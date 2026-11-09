import re
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

    for t in range(max_steps):
        # Actions executed at time step t
        current_actions = time_steps[t]
        # Map agent to their action at this time step
        agent_action = {agent: 'wait' for agent in agents}
        for action_num in current_actions:
            agent, action_desc = actions[action_num]
            agent_action[agent] = actions[action_num][1]
        # Append actions or 'wait' to each agent's plan
        for agent in agents:
            agent_plans[agent].append(agent_action[agent])

    # Print the per-agent plans
    # print(f"Left: {agent_plans['Left']}")
    # print(f"Right: {agent_plans['Right']}")
    return agent_plans['Left'], agent_plans['Right']

def plan2dict(plan):
    result = []
    for action in plan:
        match = re.match(r'(\w+)(?:\(\s*([\w\s,]*)\s*\))?$', action)
        if match:
            func_name = match.group(1)  # 提取函数名
            params_str = match.group(2)  # 可能为 None

            if params_str:
                params = [param.strip() for param in params_str.split(",") if param.strip()]
                param_dict = {f"para{i}": param for i, param in enumerate(params)}
            else:
                param_dict = {}  # 无参数时返回空字典

            result.append({"commands": func_name, "paras": param_dict})

    return result


def command_parser_pddl(command_dict):
    commands = []
    paras = []
    for index in range(len(command_dict)):
        command = command_dict[index]["commands"]
        para = command_dict[index]["paras"]
        if command == 'move_to':
            command = "move_single_to_pose"
            para = f'{{"target": "{para["para2"]}"}}'
        elif command == 'move_above':
            para = f'{{"target": "{para["para2"]}", "off_set":"up"}}'
        elif command == 'pour':
            para = '{}'
        elif command == 'wait':
            # command = "wait"
            para = '{}'
        elif command == 'grasp':
            obj_name = para['para0']
            if "cup" in obj_name:
                command = 'side_grasp'
            elif ("apple" in obj_name) or ("banana" in obj_name) or ("block" in obj_name):
                command = 'top_grasp'
            para = f'{{"obj_name": "{para["para0"]}"}}'
        elif command == 'release':
            para = f'{{"obj_name": "{para["para0"]}"}}'
        elif command == 'push':
            command = "push_to"
            para = f'{{"source_obj": "{para["para0"]}", "target_obj": "{para["para3"]}"}}'
        elif command == 'place':
            para = f'{{"source_obj": "{para["para0"]}", "target_obj": "{para["para2"]}"}}'
        elif command == 'co_move_to':
            para = f'{{"obj_name": "{para["para1"]}"}}'
        elif command == 'co_hold':
            para = f'{{"obj_name": "{para["para0"]}"}}'
        commands.append(command)
        paras.append(para)
    return commands, paras

def command_parser_direct(command_dict):
    commands = []
    paras = []
    for index in range(len(command_dict)):
        command = command_dict[index]["commands"]
        para = command_dict[index]["paras"]
        
        if command == 'move_to':
            command = "move_single_to_pose"
            para = f'{{"obj_name": "{para.get("para0", "")}", "target": "{para.get("para1", "")}"}}'
        
        elif command == 'move_above':
            para = f'{{"target": "{para.get("para1", "")}", "off_set":"up"}}'
        
        elif command in ('pour', 'wait'):
            para = '{}'
        
        elif command == 'grasp':
            obj_name = para.get("para0", "")
            if "cup" in obj_name:
                command = 'side_grasp'
            elif any(item in obj_name for item in ["apple", "banana", "block"]):
                command = 'top_grasp'
            para = f'{{"obj_name": "{obj_name}"}}'
        
        elif command == 'release':
            para = f'{{"obj_name": "{para.get("para0", "")}"}}'
        
        elif command in ['push', 'push_to']:
            command = "push_to"
            para = f'{{"source_obj": "{para.get("para0", "")}", "target_obj": "{para.get("para1", "")}"}}'
        
        elif command == 'place':
            para = f'{{"source_obj": "{para.get("para0", "")}", "target_obj": "{para.get("para1", "")}"}}'
        
        elif command == 'co_move_to':
            para = f'{{"obj_name": "{para.get("para0", "")}"}}'
        
        elif command == 'co_hold':
            para = f'{{"obj_name": "{para.get("para0", "")}"}}'
        
        commands.append(command)
        paras.append(para)
    return commands, paras