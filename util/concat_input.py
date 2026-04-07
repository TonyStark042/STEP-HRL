from util.inst import high_prompt, low_prompt, local_progress_prompt, done_prompt

# ==========================================
# High-Level Policy
# ==========================================

def get_high_obs(historical_subtasks, last_subtask_progress, current_observation):
    """ (History + Progress + Obs)"""
    hist_str = str(historical_subtasks)
    return f"Historical subtasks: {hist_str}\nLast subtask progress: {last_subtask_progress}\nCurrent observation: {current_observation}"

def get_high_prompt(task_description):
    return f"{high_prompt}{task_description}\n"

def get_high_input(task_description, historical_subtasks, last_subtask_progress, current_observation):
    """ (Prompt + Task + Context)"""
    prompt = get_high_prompt(task_description)
    context = get_high_obs(historical_subtasks, last_subtask_progress, current_observation)
    return f"{prompt}{context}"

# ==========================================
# Low-Level Policy
# ==========================================

def get_low_prompt(subtask):
    """ (Prompt + Subtask)"""
    return f"{low_prompt}Subtask: {subtask}\n"

def get_low_obs(local_progress, current_observation):
    """ (Progress + Obs)"""
    # return f"Local Progress: {local_progress}\nCurrent observation: {current_observation}"
    return f"Current Observation: {current_observation}Local Progress: {local_progress}\n"

def get_low_input(subtask, local_progress, current_observation):
    """ (Prompt + Subtask + Context)"""
    context = get_low_prompt(subtask) + get_low_obs(local_progress, current_observation)
    return f"{context}"

# ==========================================
# Local Progress Model
# ==========================================

def get_progress_prompt():
    return local_progress_prompt

def get_progress_obs(subtask, prev_local_progress, action, resulting_observation):
    """ (Progress Model Context)"""
    return f"Subtask: {subtask}\nPrevious local progress: {prev_local_progress}\nCurrent action: {action}\nResulting observation: {resulting_observation}"

def get_progress_input(subtask, prev_local_progress, action, resulting_observation):
    """ (Progress Model Input)"""
    context = get_progress_obs(subtask, prev_local_progress, action, resulting_observation)
    return f"{local_progress_prompt}{context}"

# ==========================================
# Done Model
# ==========================================

def get_done_input(subtask, local_progress, current_observation):
    context = f"Subtask: {subtask}\nPrevious progress: {local_progress}\nCurrent observation: {current_observation}"
    return f"{done_prompt}{context}"