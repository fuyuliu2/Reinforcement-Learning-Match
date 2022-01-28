from typing import Dict
import torch
import user_states


def get_next_state(inputs : Dict[str, torch.Tensor]) -> torch.Tensor:
    # Extract variables from inputs
    # add policy action for additional states
    states = inputs["state"]
    user_action = inputs["action_2"]
    day_order_num = user_action[..., 0]
    day_avg_fee   = user_action[..., 1]
    # Construct next_states array with currrent state's shape and device
    next_states = states.new_empty(states.shape)
    user_states.get_next_state(states, day_order_num, day_avg_fee, next_states)
    return next_states



