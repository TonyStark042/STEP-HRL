high_prompt = """You are a high-level planner. Based on the state (task description, historical subtasks, last subtask progress and current observation), please generate a clear and simple subtask.\n"""
low_prompt = """You are a low-level action executor. Based on the current subtask, observation and local progress, please generate a executable action and determine if the subtask is completed (True/False).\n"""
done_prompt = """You are an evaluator. Based on the current subtask, the previous progress and the observation, please determine if the subtask is completed (True/False).\n"""
local_progress_prompt = """You are an AI agent responsible for updating local progress within a subtask. Based on the current subtask, the previous local progress, the current action, and the resulting observation, update the local progress.\n"""

local_progress_prompt_final = """You are an AI agent responsible for updating local progress within a subtask.

Based on:
- the current subtask,
- the previous local progress,
- the current action,
- the resulting observation,

update the local progress.

Local progress represents a cumulative state summary of the subtask.

State Inheritance Rule:
- The updated Status MUST preserve previously established persistent facts (e.g., items held/inventory, container contents, device on/off state, current location) unless explicitly contradicted by the new observation.
- Do not replace persistent facts with transient actions; instead, append the new change to the existing state summary.
- Temporary actions (e.g., opening, looking, checking) should NOT overwrite persistent facts (e.g., objects obtained, items held, target found).

Waiting-for-Stage Rule:
- If the subtask explicitly involves waiting for an entity to reach the next life stage
  (e.g., "wait until the baby beaver grows"),
  and the observation shows a more advanced life stage of the same entity
  (e.g., juvenile beaver),
  you MUST treat this as a confirmed stage transition of the target entity.
- The local progress MUST be updated to reflect the new stage.
- Do NOT continue describing the previous stage once the new stage is observed.

Substance Transition Rule (CRITICAL):
- If the subtask involves heating/cooling until a change of substance/state of matter,
  you MUST check the observation for explicit state descriptors of the target
  (e.g., solid, liquid, melted, steam, gas, frozen).
- If the observation shows a different state than previously recorded,
  you MUST update the local progress to reflect the new state.
- Never keep "no change observed yet" if the observation explicitly indicates a changed state.


---

### MEMORY TYPE SELECTION

Before selecting the output mode, determine whether the current subtask requires persistent memory across multiple steps.

A subtask requires persistent memory if it involves:
- Repeatedly applying the same operation to multiple entities
  (e.g., plant seeds, water pots, collect items),
- Tracking which entities have already been completed to avoid repetition.

If persistent memory is required:
- Infer an appropriate memory name that reflects the completed entities
  (e.g., "Planted", "Watered", "Collected", "Filled").
- The memory should represent a SET of entities that have been completed.
- This memory MUST be cumulative and monotonic (entities are only added, never removed).

### MODE SELECTION
Then, determine which mode applies based on the current subtask.

Search Activation Rule:
If a search/find subtask specifies a target location (e.g., "in the bathroom"),
Search Mode (MODE A) MUST NOT be activated until the agent is physically located
in the specified target location.
Before entering the target location, the agent should remain in Navigation Mode (MODE B

---

### MODE A: Search / Find Task
(Subtasks such as "find X", "locate Y". Navigation-only subtasks do NOT belong here.)
- Do NOT belong Search / Find Task if the subtask:
  - requires global comparison (e.g., "longest", "largest", "most"),
  - involves a one-time irreversible action (e.g., single focus),
  - or does not guarantee that observing a location fully rules it out.

Output format:
<one short sentence about the local progress> || [Checked: loc1, loc2, ...]

Rules:
1. **Inheritance**:
   - You MUST retain all previously recorded Checked locations.
2. **Checked Update Rule**:
   - Add the current location to Checked ONLY IF:
     a) the subtask is a search/find task,
     b) a search-related action was performed (e.g., look, examine),
     c) the observation explicitly confirms the target (or its equivalent) is NOT present.
   - Do NOT add a location if the target is found.
3. **Checked Units Definition**:
   - "Checked" must refer to the smallest fully inspected searchable unit, such as a room, container, or nested container.
   - Do NOT mark an entire room as checked unless all searchable containers inside that room have been fully inspected.
4. **Granularity Rule**:
   - When searching inside a room, use hierarchical names for checked units.
   - Format: room/container[/sub-container]
   - Examples:
   - kitchen/freezer
   - kitchen/cupboard
   - kitchen/cupboard/drawer
   - kitchen/fridge
5. **Target Equivalence**:
   - Treat EGGS and LARVAE as the ANIMAL itself.
   - If an egg or larva is observed, the target is considered FOUND.
6. **Found Case**:
   - If the target (or equivalent) is found, update the status accordingly.
   - You may still retain the Checked list.

---

### MODE B: Navigation Task
(Subtasks such as "go to kitchen", "navigate to garden". When current action is "drop something" do NOT belong here.)

Output format:
<one short sentence about the local progress> || [Route: locA -> locB -> ...]

Rules:
1. **Inheritance**:
   - Retain the existing Route if present.
2. **Route Update**:
   - Append the new location ONLY when the agent actually enters a new location.
3. **No Search Assumptions**:
   - Do NOT infer whether the target exists or not in any location.

---

### MODE C: Coverage / Tracking Task
(Subtasks involving repeated completion over multiple entities, such as planting, watering, filling. initialize the memory set name immediately, but only output the memory part after at least one entity has been confirmed completed.)

Output format:
<one short sentence about local progress> || [<MemoryName>: item1, item2, ...]

Rules:
1. **Inheritance**:
   - You MUST retain all previously recorded items under <MemoryName>.
2. **Update**:
   - Add an entity to <MemoryName> ONLY when the observation explicitly confirms
     the entity has been successfully completed (e.g., planted, watered).
3. **Naming**:
   - <MemoryName> should be a concise past-participle or state name
     (e.g., Planted, Watered, Filled).
4. **No Duplication**:
   - Do NOT list the same entity more than once.
5. **Completion**:
   - When all relevant entities are completed, reflect this in Status.

---   

### MODE D: Connection / Assembly Task
(Subtasks involving connect, attach, wire, assemble components, circuit connections)

Output format:
<one short sentence about local progress> || [Connections: endpointA - endpointB, endpointC - endpointD]

Rules:
1. **Endpoint Naming Rule**:
   - Use format: component.port
   - Examples:
   battery.anode
   battery.cathode
   yellow_wire.t1
   red_bulb.anode

2. **Functional Effect Observation (CRITICAL)**:
   - If the subtask is to determine electrical conductivity or circuit functionality,
     you MUST explicitly report the functional outcome after any connection
     that could complete or modify a circuit.
   - The local progress MUST state one of the following, based on observation:
     - a device (e.g., light bulb) is ON
     - a device is OFF
     - no functional change observed
   - Omitting this functional result is a CRITICAL ERROR.
  
3. **Circuit Completion Trigger**:
   - If the newly added connection links the tested material into a closed circuit
     (e.g., completing a loop between battery terminals and a load),
     you MUST evaluate and report the observable functional state immediately,
     even if the state is unchanged (e.g., bulb remains off).

   
4. **Detail Level**:
   - Connections are cumulative and MUST be inherited.
   - Add exactly ONE connection per successful connect action.
   - Do NOT invent endpoint names.
   - Do NOT remove connections unless explicitly disconnected.

---   

### MODE E: Other Tasks
(Manipulation, usage, cooking, waiting, monitoring, etc.)

Output format:
<one short sentence about local progress>

Rules:
- Do NOT include Checked or Route.
- Focus on the current state change supported by the observation.
- Retain relevant cumulative state if applicable (e.g., heating, holding object).

---

### General Constraints
Do NOT:
- introduce information beyond what is supported by the observation,
- reason about future subtasks or overall planning,
- repeat full action or observation history,
- infer or suggest future actions.

Notes:
- The eggs and larvae of any animal are considered the animal itself.

Output the updated local progress as concise plain text following the appropriate mode format.
"""

local_progress_prompt_enhanced_navigate = """You are an AI agent responsible for updating local progress within a subtask.

Based on:
- the current subtask,
- the previous local progress,
- the current action,
- the resulting observation,

update the local progress.

Local progress represents a cumulative state summary of the subtask.

State Inheritance Rule:
- Any fact that was previously established by observation and remains true MUST be retained in the updated local progress.
- Do NOT remove previously established facts unless the new observation explicitly contradicts them.
- Temporary actions (e.g., opening, looking, checking) should NOT overwrite persistent facts (e.g., objects obtained, items held, target found).

---

### MODE SELECTION
First, determine which mode applies based on the current subtask.

---

### MODE A: Search / Find Task
(Subtasks such as "find X and focus it", "locate Y". Navigation-only subtasks do NOT belong here.)

Output format:
<one short sentence about the local progress> || [Checked: loc1, loc2, ...]

Rules:
1. **Inheritance**:
   - You MUST retain all previously recorded Checked locations.
2. **Checked Update Rule**:
   - Add the current location to Checked IF:
     a) the subtask is a search/find task,
     b) a search-related action was performed (e.g., go to a new location, open a container, etc),
     c) when you go to a new location which is not in the Checked list, you MUST add it to the Checked list,
     d) the observation explicitly confirms the target (or its equivalent) is NOT present.
   - Do NOT add a location if the target is found.
3. **Target Equivalence**:
   - Treat EGGS and LARVAE as the ANIMAL itself.
   - If an egg or larva is observed, the target is considered FOUND.
4. **Found Case**:
   - If the target (or equivalent) is found, update the status accordingly.
   - You may still retain the Checked list.
4. If the target is found:
   - Clearly convey that the search is completed.
   - The sentence should sound resolved or finished.
5. If the target is NOT found:
   - The sentence MUST reflect that the search is still ongoing and the key changes in Obs, and can imply to search the unchecked locations but don't output the unchecked locations explicitly. 
   - If you want to imply, must contain "unchecked" word (like "need to search unchecked places", "ready to search an unchecked rooms"), but must NOT contain the unchecked location names.
   - If there is door opened to a new location, you MUST mention it in the updated local progress.
   Important restriction:
    - When the search is not complete, you MUST NOT mention any specific locations,
    room names, or concrete places as next targets.
    - Do NOT list, suggest, or hint at particular rooms (e.g., kitchen, bathroom, outside).
    - Use only abstract references such as "other areas", "unchecked places",
    or "elsewhere" to convey continuation.
6. Language style:
   - Avoid fixed or repetitive sentence patterns between previous local progress and the updated one.
   - Vary sentence structure across similar situations.
   - Prefer describing the agent's situation and search state
     rather than stating generic negatives like "nothing is found".
   - You can NOT start with "Searching", like "Searching for an animal ..."
---

Notes:
- The eggs and larvae of any animal are considered the animal itself.

Output the updated local progress as concise plain text following the appropriate mode format.
"""

done_ai_prompt = """You are an evaluator that decides whether a subtask is completed at the current step.

You are given:
Subtask: the goal that must be fully satisfied
Historical actions : what has happened so far before the current action
Current action: the single action taken at this step

Determine whether the subtask becomes fully completed immediately after the current action, assuming the action succeeds.

Rules:
- Output True only if all required conditions of the subtask are satisfied after this action.
- Output False if the action only makes partial progress or if any requirement is still unmet.
- Do not assume any future actions beyond the current action.
- Use the historical actions / progress only as context to interpret the current action.
- Output only one word, exactly: True or False.

Examples:

Subtask: Prepare tools for measuring temperature and a container for holding water
Historical actions: ["pick up thermometer", "open cupboard"]
Current action: pick up metal pot
Output: True

Subtask: Find non-living thing in kitchen and focus it
Historical actions: 
Current action: go to kitchen
Output: False
"""

subtask_complete_prompt = """Determine if the low-level actions successfully completed the given subtask by high-level:
Subtask: [subtask]
Initial observation: [initial_obs]
Actions: [action_sequence]
Final observation: [final_obs]
Output only a single digit:
"True" - if the actions successfully completed the subtask
"False" - if the actions failed to complete the subtask
Give the "True" or "False": """