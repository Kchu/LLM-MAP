import subprocess, json, re
import time
global RIGHT_FINGER_STAT
global LEFT_FINGER_STAT
RIGHT_FINGER_STAT = LEFT_FINGER_STAT = 'Open'
global LEFT_HAND_STAT
global RIGHT_HAND_STAT
LEFT_HAND_STAT = RIGHT_HAND_STAT = 'Vertical'

# Filter results from the stdouts
##############################################################################
def extract_parts(side, text):
    # Pattern to match the sentence
    sentence_pattern = r"RIGHT_RESULT: (.*?)\. RIGHT_HAND:" if side == "right" else r"LEFT_RESULT: (.*?)\. LEFT_HAND:"
    # Pattern to match the RIGHT_HAND value
    hand_pattern = r"RIGHT_HAND: (\w+)" if side == "right" else r"LEFT_HAND: (\w+)"
    # Pattern to match the RIGHT_FINGER value
    finger_pattern = r"RIGHT_FINGER: (\w+)" if side == "right" else r"LEFT_FINGER: (\w+)"

    # Extract the sentence
    sentence_match = re.search(sentence_pattern, text)
    sentence = sentence_match.group(1) if sentence_match else None

    # Extract the RIGHT_HAND value
    hand_match = re.search(hand_pattern, text)
    hand = hand_match.group(1) if hand_match else None

    # Extract the RIGHT_FINGER value
    finger_match = re.search(finger_pattern, text)
    finger = finger_match.group(1) if finger_match else None

    return sentence, hand, finger
##############################################################################

def plan2dict(plan):
    result = []
    for action in plan:
        if action=="wait": action = "wait()"
        # match = re.match(r'(\w+)(?:\(\s*([\w\s,]*)\s*\))?$', action)
        pattern = r'(\w+)\(\s*([\'"\w\s,]*)\s*\)?$'
        match = re.match(pattern, action)
        if match:
            func_name = match.group(1)
            params_str = match.group(2)

            if params_str:
                params = [param.strip().strip("'\"") for param in params_str.split(",") if param.strip()]
                param_dict = {f"para{i}": param for i, param in enumerate(params)}
            else:
                param_dict = {}
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

def nicol_control(file_path, is_pddl, task):
    RIGHT_FINGER_STAT = LEFT_FINGER_STAT = 'Open'
    LEFT_HAND_STAT = RIGHT_HAND_STAT = 'Vertical'
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    left_dict, right_dict = plan2dict(data["Left"]), plan2dict(data["Right"])
    if is_pddl:
        left_commands, left_paras = command_parser_pddl(left_dict)
        right_commands, right_paras = command_parser_pddl(right_dict)
    else:
        left_commands, left_paras = command_parser_direct(left_dict)
        right_commands, right_paras = command_parser_direct(right_dict)
    # print("\n Executing the plan... \n")
    for index in range(len(left_commands)):
        left_command = left_commands[index]
        left_para = left_paras[index]
        right_command = right_commands[index]
        right_para = right_paras[index]
        # make sure both subpresses are executing hold_up_single
        if "co_hold" in left_command: right_command, right_para = left_command, left_para
        if "co_hold" in right_command: left_command, left_para = right_command, right_para
        # make sure only one subpress is executing move_both_to_poses
        if "wait" in left_command: 
            left_para = '{}'
            head_process = subprocess.Popen(['python', './NICOL/nicol_executor.py', '--command', "look_at", '--para', '{"obj_name":"right"}'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        elif "wait" in right_command: 
            right_para = '{}'
            head_process = subprocess.Popen(['python', './NICOL/nicol_executor.py', '--command', "look_at", '--para', '{"obj_name":"left"}'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        else:
            head_process = subprocess.Popen(['python', './NICOL/nicol_executor.py', '--command', "look_at", '--para', '{"obj_name":"center"}'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if "co_move_to" in left_command: right_command, right_para = "wait", '{}'
        elif "co_move_to" in right_command: left_command, left_para = "wait", '{}'
        if left_command == right_command and left_command in ["move_to", "move_above"] and left_para == right_para and 'block' in left_para:
            left_result = right_result = "failed. You can not move two objects to the same place!"
        else:
            print(f"Left: {left_command}, {left_para} Right: {right_command}, {right_para}")
            left_proc = subprocess.Popen(['python', './NICOL/nicol_executor.py', '--side', 'left', '--command', left_command, '--para', left_para, f"--hand_state={LEFT_HAND_STAT}", f"--finger_state={LEFT_FINGER_STAT}"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            right_proc = subprocess.Popen(['python', './NICOL/nicol_executor.py', '--side', 'right', '--command', right_command, '--para', right_para, f"--hand_state={RIGHT_HAND_STAT}", f"--finger_state={RIGHT_FINGER_STAT}"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            l_stdout, l_error = left_proc.communicate()
            r_stdout, r_error = right_proc.communicate()
            h_stdout, h_stderr = head_process.communicate()
            left_result, left_hand, left_finger = extract_parts("left", l_stdout)
            right_result, right_hand, right_finger = extract_parts("right", r_stdout)
            print(f"Left result: ", left_result, left_hand, left_finger )
            print(f"Right result:", right_result, right_hand, right_finger )
            LEFT_HAND_STAT, LEFT_FINGER_STAT = left_hand, left_finger
            RIGHT_HAND_STAT, RIGHT_FINGER_STAT = right_hand, right_finger
        if left_result is None or right_result is None or "failed" in str(left_result) or "failed" in str(right_result):
            if task.check_success():
                print("The task is successfully accomplished!")
                return True, index+1
            else:
                # print("Left result: ", left_result, "Right result:", right_result)
                print(f"The task is failed starting from step {index+1}!")
                return False, index+1
    head_process = subprocess.Popen(['python', './NICOL/nicol_executor.py', '--command', "look_at", '--para', '{"obj_name":"human"}'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    h_stdout, h_stderr = head_process.communicate()
    time.sleep(4)
    if task.check_success():
        print("The task is successfully accomplished!")
        return True, len(left_commands)
    else:
        print(f"The task is failed after the whole {len(left_commands)} steps!")
        return False, len(left_commands)