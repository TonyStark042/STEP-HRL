from openai import OpenAI

client = OpenAI(
    api_key="sk---",
    base_url="https://api.deepseek.com",
)

def llm(prompt):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    response = client.chat.completions.create(
      model="deepseek-chat",
      messages = messages,
    #   prompt=prompt,
      temperature=1,
      max_tokens=1024,
      top_p=1,
      frequency_penalty=0.0,
      presence_penalty=0.0,
    #   stop=stop
    )
    return response.choices[0].message.content

def apichatgpt_givenprompt(prompt, lm="deepseek-chat"):

    completions = client.chat.completions.create(
        model=lm,
        n=1,
        messages=prompt,
        temperature=0.0,
    )
    
    instructions = completions.choices[0].message.content

    return instructions

desc = instruction = """
    You are an agent operating inside a simulated environment.
    You will be given a task along with its description.
    Based on what you observe in the environment and the items in your inventory, you must suggest the best possible actions to complete the task.

    Only use valid actions and valid objects. If you know what is around you, propose the next appropriate action.

    You are allowed to perform the following actions:

    - open OBJ: open a container  
    - close OBJ: close a container  
    - activate OBJ: activate a device  
    - deactivate OBJ: deactivate a device  
    - connect OBJ to OBJ: connect electrical components  
    - disconnect OBJ: disconnect electrical components  
    - use OBJ [on OBJ]: use a device or item  
    - look around: describe the current room  
    - look at OBJ: describe an object in detail  
    - look in OBJ: describe the contents of a container  
    - read OBJ: read a note or book  
    - move OBJ to OBJ: move an object into another container  
    - pick up OBJ: add an object to the inventory  
    - put down OBJ: drop an inventory item  
    - pour OBJ into OBJ: pour a liquid into a container  
    - dunk OBJ into OBJ: dunk a container into a liquid  
    - mix OBJ: chemically mix a container  
    - go to LOC: move to a new location  
    - teleport to LOC: teleport to a specific room  
    - eat OBJ: consume a food item  
    - flush OBJ: flush a toilet  
    - focus on OBJ: signal intent on a task object  
    - wait [DURATION]: take no action for some duration  
    - task: describe the current task  
    - inventory: list the agent’s inventory  

    Definitions:  
    - OBJ = objects  
    - LOC = location  

    There are 9 available locations themed around a house: kitchen, bathroom, workshop, art studio, greenhouse, outside, bedroom, living room, foundry.
    """

def create_initialprompt_start(sample1):
    """This function create samples for the prompting using the samples."""
    task_desc1 = "Task Description:\nYour task is to boil water. For compounds without a boiling point, combusting the substance is also acceptable. First, focus on the substance. Then, take actions that will cause it to change its state of matter."
    path1 = [
        "open door to kitchen",
        "go to kitchen",
        "pick up thermometer",
        "open cupboard",
        "pick up metal pot",
        "move metal pot to sink",
        "activate sink",
        "deactivate sink",
        "pick up metal pot",
        "focus on substance in metal pot",
        "pour metal pot into metal pot",
        "pick up metal pot",
        "move metal pot to stove",
        "activate stove",
        "examine substance in metal pot",
        "use thermometer in inventory on substance in metal pot",
        "examine substance in metal pot",
        "use thermometer in inventory on substance in metal pot",
        "examine substance in metal pot",
        "use thermometer in inventory on substance in metal pot",
        "examine substance in metal pot",
        "use thermometer in inventory on substance in metal pot",
        "examine substance in metal pot",
        "use thermometer in inventory on substance in metal pot",
        "examine substance in metal pot",
        "use thermometer in inventory on substance in metal pot",
        "examine substance in metal pot",
        "use thermometer in inventory on substance in metal pot",
        "examine substance in metal pot",
        "use thermometer in inventory on substance in metal pot",
        "examine substance in metal pot",
        "use thermometer in inventory on substance in metal pot",
        "examine substance in metal pot",
        "use thermometer in inventory on substance in metal pot",
        "examine steam",
        "use thermometer in inventory on steam",
        "wait1",
        "wait1",
    ]
    goalpath1 = ""
    for step in path1:
        goalpath1 += step + ", "

    # Construct example instances
    messages = [
        {"role": "system", "content": "You are a helpful assistant." + desc},
        {
            "role": "user",
            "content": task_desc1
            + " Here is the gold path to achieve to the goal:"
            + goalpath1
            + ". Based on the given gold path, provide me with the functional format of high-level sub-tasks to complete this task and their correspondings actions.",
        },
        {
            "role": "assistant",
            "content": "1- navigate_to(kitchen) : {'open door to kitchen', 'go to kitchen'} 2- pick_up(thermometer):{'pick up thermometer'} 3- find(metal pot):{'open cupboard', 'pick up metal pot'} 4- Fill(metal pot, water): {'move metal pot to sink', 'activate sink', 'deactivate sink', 'pick up metal pot'} 5- Focus_on(substance in metal pot):{'focus on substance in metal pot'} 6- heat(water, metal pot): {'pour metal pot into metal pot', 'pick up metal pot', 'move metal pot to stove', 'activate stove'} 7- Monitor_temperature(metal pot, boil): {'examine substance in metal pot', 'use thermometer in inventory on substance in metal pot', 'examine substance in metal pot', 'use thermometer in inventory on substance in metal pot', 'examine substance in metal pot', 'use thermometer in inventory on substance in metal pot', 'examine substance in metal pot', 'use thermometer in inventory on substance in metal pot', 'examine substance in metal pot', 'use thermometer in inventory on substance in metal pot', 'examine substance in metal pot', 'use thermometer in inventory on substance in metal pot', 'examine substance in metal pot', 'use thermometer in inventory on substance in metal pot', 'examine substance in metal pot', 'use thermometer in inventory on substance in metal pot', 'examine substance in metal pot', 'use thermometer in inventory on substance in metal pot'} 8- chek(steam): {'examine steam', 'use thermometer in inventory on steam'} 9- wait(2): {'wait1', 'wait1'}",
        }
    ]

    # prompt to generate subtask given the new gold path
    path_sample = sample1["gold_path"]
    goalpath_sample = ""
    for step in path_sample:
        goalpath_sample += step + ", "

    messages.append(
        {
            "role": "user",
            "content": sample1["task_desc"]
            + " Here is the gold path to achieve to the goal: "
            + goalpath_sample
            + ". Based on the given gold path, provide me with the functional format of high-level sub-tasks to complete this task and their correspondings actions. ",
        }
    )

    ## Call GPT4 or ChatGPT
    GPT4output = apichatgpt_givenprompt(messages)
    messages.append({"role": "assistant", "content": GPT4output})

    return GPT4output



from scienceworld import ScienceWorldEnv
import json
import os

if __name__ == "__main__":
    TASK_NUM = 30
    benchmark = "scienceworld"

    env = ScienceWorldEnv("", envStepLimit=1000)
    task_names = env.getTaskNames()
    data_type = "train"

    for task_id in range(12, 13):
        env.load(task_names[task_id], generateGoldPath=True)
        if data_type == "train":
            vari_ids = env.getVariationsTrain()
        elif data_type == "dev":
            vari_ids = env.getVariationsDev()
        elif data_type == "test":
            vari_ids = env.getVariationsTest()
        save_path = f"dataset/{benchmark}/subtask_data/task{task_id}/"
        os.makedirs(save_path, exist_ok=True)
        for vari in vari_ids:
            with open(f'dataset/{benchmark}/expert_path/task{task_id}/variation{vari}.json', 'r') as json_file:
                raw_data = json.load(json_file)

            sample = {
                "task_desc": raw_data['task_description'],
                "gold_path": raw_data['action']
            }
            raw_data['subtask'] = create_initialprompt_start(sample)
            
            with open(save_path+f'variation{vari}.json', 'w') as json_file:
                json.dump(raw_data, json_file, indent=4)
                print(task_id, vari,"   done")


        # exit()