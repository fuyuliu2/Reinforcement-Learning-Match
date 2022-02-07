from typing import Dict
import torch
import user_states
import get_sample_data


def get_next_state(inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    # Extract variables from inputs
    # add policy action for additional states
    states = inputs["state"]
    user_action = inputs["action_2"]
    day_order_num = user_action[..., 0]
    day_avg_fee = user_action[..., 1]
    coupon_action = inputs['action_1']
    coupon_num = coupon_action[..., 0]
    coupon_discount = coupon_action[..., 1]
    # Construct next_states array with currrent state's shape and device
    next_states = states.new_empty(states.shape)
    user_states.get_next_state_torch(states, day_order_num, day_avg_fee, coupon_num, coupon_discount, next_states)

    return next_states


def get_coupon_action(inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    initial_coupon = inputs['action_1']
    modified_coupon = initial_coupon.new_empty(initial_coupon.shape)
    modified_coupon[..., 0] = get_sample_data.get_sample(initial_coupon[..., 0], 'day_deliver_coupon_num', 0.6)
    modified_coupon[..., 1] = initial_coupon[..., 1]
    return modified_coupon


def get_user_action(inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    initial_user = inputs['action_2']
    modified_user = initial_user.new_empty(initial_user.shape)
    modified_user[..., 0] = get_sample_data.get_sample(initial_user[..., 0], 'day_order_num', 0.6)
    modified_user[..., 1] = initial_user[..., 1]
    return modified_user


"""
1.get_next_state :user_action = inputs["action_2"]----user_action = inputs["user"]
2.add get_coupon_action
3.add get_user_action
"""



