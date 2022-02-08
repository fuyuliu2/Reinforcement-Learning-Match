import numpy as np

def get_state_names():
    return ["total_num", "average_num", "average_fee"]


def get_next_state(states, day_order_num, day_average_fee, next_states):
    # If you define day_order_num to be continuous instead of discrete/category, apply round function here.
    day_order_num = day_order_num.clip(0, 6).round()
    day_average_fee = day_average_fee.clip(0.0, 100.0)
    # Rules on the user action: if either action is 0 (order num or fee), the other action should also be 0.
    day_order_num[day_average_fee <= 0.0] = 0
    day_average_fee[day_order_num <= 0] = 0.0
    # We compute the days accumulated for each user's state by dividing the total order num with average order num
    accumulated_days = states[..., 0] / states[..., 1]
    accumulated_days[states[..., 1] == 0.0] = 0.0
    # Compute next state
    next_states[..., 0] = states[..., 0] + day_order_num # Total num
    next_states[..., 1] = states[..., 1] + 1 / (accumulated_days + 1) * (day_order_num - states[..., 1]) # Average order num
    next_states[..., 2] = states[..., 2] + 1 / (accumulated_days + 1) * (day_average_fee - states[..., 2]) # Average order fee across days
    return next_states


def states_to_observation(states: np.ndarray, day_total_order_num: int=0, day_roi: float=0.0):
    """Reduce the two-dimensional sequence of states of all users to a state of a user community
        A naive approach is adopted: mean, standard deviation, maximum and minimum values are calculated separately for each dimension.
        Additionly, we add day_total_order_num and day_roi.
    Args:
        states(np.ndarray): A two-dimensional array containing individual states for each user
        day_total_order_num(int): The total order number of the users in one day
        day_roi(float): The day ROI of the users
    Return:
        The states of a user community (np.array)
    """
    assert len(states.shape) == 2
    mean_obs = np.mean(states, axis=0)
    std_obs = np.std(states, axis=0)
    max_obs = np.max(states, axis=0)
    min_obs = np.min(states, axis=0)
    day_total_order_num, day_roi = np.array([day_total_order_num]), np.array([day_roi])
    return np.concatenate([mean_obs, std_obs, max_obs, min_obs, day_total_order_num, day_roi], 0)


