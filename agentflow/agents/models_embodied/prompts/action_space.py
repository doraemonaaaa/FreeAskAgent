def action_space(agent):
    return f'''
## Action Space for {agent}
By this you can judge whether planner output good command to complete the subgoal.
### <Forward(n)>  # Move forward n meter
### <TurnLeft(n)>  # Turn left n degree
### <TurnRight(n)>  # Turn Right n degree
### <TurnAround()>  # Turn around for 180 degree
### <Ask(text)>
- **Pre-condition**: MUST ONLY be used if a Human is visible AND distance < 2.0 meters.
- **Usage**: Request social or goal-related info (e.g., "Where is the kitchen?", "Are you the person I'm looking for?").
- **Example**: `<Ask("Excuse me, could you tell me where the apple is?")>`
### <Wait(t)>
- **Description**: Pause execution for a specific duration.
- **Parameters**: `t` [Float/Int] in seconds.
- **Example**: `<Wait(5)>`
'''