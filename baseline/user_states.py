import numpy as np
import torch
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA


def get_state_names():
    return ["total_day", "use_discount_day", "total_num", "average_num", "average_fee", "coupon_use", "avg_used_discount",
            "order_day", "coupon_order", "discount_fee"]
    # return ["total_num", "average_num", "average_fee"]
# add "coupon_use", "avg_used_discount", "order_day", "coupon_order", "discount_fee"


def get_next_state(states, day_order_num, day_average_fee, coupon_num, coupon_discount, next_states):
    # If you define day_order_num to be continuous instead of discrete/category, apply round function here.
    day_order_num = day_order_num.clip(0, 6).round()
    day_average_fee = day_average_fee.clip(0.0, 100.0)
    # ** normalize coupon action **
    coupon_num = coupon_num.clip(0, 5).round()
    coupon_discount = (coupon_discount * 20).round() / 20
    # Rules on the user action: if either action is 0 (order num or fee), the other action should also be 0.
    day_order_num[day_average_fee <= 0.0] = 0
    day_average_fee[day_order_num <= 0] = 0.0
    # We compute the days accumulated for each user's state by dividing the total order num with average order num
    accumulated_days = states[..., 2] / states[..., 3]
    accumulated_days[states[..., 3] == 0.0] = 0.0

    # Compute next state
    next_states[..., 0] = states[..., 0] + 1  # num of total day
    next_states[..., 1] = states[..., 1] + (day_order_num > 0.0) * (coupon_num > 0.0)  # num of use_discount_day
    next_states[..., 2] = states[..., 2] + day_order_num  # Total num
    next_states[..., 3] = states[..., 3] + 1 / (accumulated_days + 1) * (day_order_num - states[..., 3]) * (day_order_num > 0)  # Average order num
    next_states[..., 4] = states[..., 4] + 1 / (accumulated_days + 1) * (day_average_fee - states[..., 4]) * (day_average_fee > 0)  # Average order fee across days
    next_states[..., 5] = (states[..., 2] + day_order_num) / (states[..., 2] / np.maximum(states[..., 5], 1) + np.maximum(coupon_num, 1))  # coupon_use
    # next_states[..., 4] = states[..., 4] + (coupon_discount * (day_order_num > 0.0) * (coupon_num > 0.0) -
    #                                         states[..., 4]) / (accumulated_days + 1)  # avg_used_discount
    used_discount = coupon_discount * (day_order_num > 0.0) * (coupon_num > 0.0)
    next_states[..., 6] = states[..., 6] + (used_discount - states[..., 6]) * (used_discount > 0.0) / np.maximum(next_states[..., 1], 1)  # avg_used_discount
    next_states[..., 7] = accumulated_days / next_states[..., 0] + (day_order_num > 0.0) / next_states[..., 0] # ratio of order day and total day
    day_coupon_used_num = np.minimum(day_order_num, coupon_num)
    # next_states[..., 6] = (states[..., 6] * states[..., 1] + day_coupon_used_num) / np.maximum((states[..., 0] + day_order_num), 0.1)  # coupon_order
    # next_states[..., 6] = states[..., 6] if day_order_num == 0 else (states[..., 6] * states[..., 1] + day_coupon_used_num) / np.maximum((states[..., 0] + day_order_num), 0.1)
    total_num = np.maximum((states[..., 2] + day_order_num), 0.1)
    next_states[..., 8] = states[..., 8] + (states[..., 8] * (states[..., 3] - total_num) + day_coupon_used_num ) * (day_order_num > 0.0) / total_num  # coupon_order
    # cost = (1 - coupon_discount) * day_coupon_used_num * day_average_fee
    # next_states[..., 9] = (states[..., 9] * accumulated_days * states[..., 4] + cost) / np.maximum((accumulated_days + 1) * next_states[..., 4], 0.1)  # discount_fee
    next_states[..., 9] = next_states[..., 1] / next_states[..., 0]  # use_discount_day / total day
    # next_states[..., 10] = accumulated_days
    return next_states


def get_next_state_torch(states, day_order_num, day_average_fee, coupon_num, coupon_discount, next_states):
    # If you define day_order_num to be continuous instead of discrete/category, apply round function here.
    day_order_num = day_order_num.clip(0, 6).round()
    day_average_fee = day_average_fee.clip(0.0, 100.0)
    # ** normalize coupon action **
    coupon_num = coupon_num.clip(0, 5).round()
    coupon_discount = (coupon_discount * 20).round() / 20
    # Rules on the user action: if either action is 0 (order num or fee), the other action should also be 0.
    day_order_num[day_average_fee <= 0.0] = 0
    day_average_fee[day_order_num <= 0] = 0.0
    # We compute the days accumulated for each user's state by dividing the total order num with average order num
    accumulated_days = states[..., 0] / states[..., 1]
    accumulated_days[states[..., 1] == 0.0] = 0.0
    device = torch.device('cuda')
    used_discount = coupon_discount * (day_order_num > 0.0) * (coupon_num > 0.0)
    day_coupon_used_num = torch.minimum(day_order_num, coupon_num)
    total_num = states[..., 2] + day_order_num
    total_num_ = torch.maximum(total_num.to(device), torch.ones(total_num.shape).to(device))
    num = states[..., 3].to(device) - total_num_.to(device)
    history_coupon = states[..., 2].to(device) / (torch.maximum(states[..., 5].to(device), torch.ones(states[..., 5].shape).to(device)))

    # Compute next state
    # one = torch.ones(states[..., 5].shape).to(torch.device('cpu'))
    next_states[..., 0] = states[..., 0] + 1  # num of total day
    next_states[..., 1] = states[..., 1] + (day_order_num > 0.0) * (coupon_num > 0.0)  # num of use_discount_day
    next_states[..., 2] = states[..., 2] + day_order_num  # Total num
    next_states[..., 3] = states[..., 3] + 1 / (accumulated_days + 1) * (day_order_num - states[..., 3]) * (day_order_num > 0) # Average order num
    next_states[..., 4] = states[..., 4] + 1 / (accumulated_days + 1) * (day_average_fee - states[..., 4]) * (day_average_fee > 0) # Average order fee across days
    next_states[..., 5] = next_states[..., 2].to(device) / (
                history_coupon.to(device) + torch.maximum(coupon_num.to(device), torch.ones(states[..., 5].shape).to(device)))  # coupon_use
    discount_day = torch.maximum(next_states[..., 1].to(device), torch.ones(states[..., 6].shape).to(device))
    next_states[..., 6] = states[..., 6].to(device) + (used_discount.to(device) - states[..., 6].to(device)) * (
                used_discount > 0.0).to(device) / discount_day.to(device)  # avg_used_discount
    next_states[..., 7] = accumulated_days / next_states[..., 0] + (day_order_num > 0.0) / next_states[
        ..., 0]  # ratio of order day and total day
    next_states[..., 8] = states[..., 8].to(device) + (states[..., 8].to(device) * num.to(device) + day_coupon_used_num.to(device)) * (
                day_order_num > 0.0).to(device) / total_num_.to(device)  # coupon_order
    next_states[..., 9] = next_states[..., 1] / next_states[..., 0]

    return next_states


def states_to_observation(states: np.ndarray, day_total_order_num: int = 0, day_roi: float = 0.0):
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
    index = np.load('user_classification.npy', mmap_mode='r').astype(int)
    # delete some lazy users
    activate_states = np.delete(states, index, 0)
    activate_states = np.delete(activate_states, [0, 1], 1)
    # pca
    pca = PCA(n_components=4).fit_transform(activate_states)
    # minibatchkmeans
    # k_means = MiniBatchKMeans(n_clusters=3, init='k-means++', max_iter=500, batch_size=100).fit(pca)
    lower_obs = np.quantile(pca, 0.25, interpolation='lower', axis=0)
    higher_obs = np.quantile(pca, 0.75, interpolation='higher', axis=0)
    median_obs = np.quantile(pca, 0.5, axis=0)
    mean_obs = np.mean(pca, axis=0)
    std_obs = np.std(pca, axis=0)
    max_obs = np.max(pca, axis=0)
    min_obs = np.min(pca, axis=0)
    day_total_order_num, day_roi = np.array([day_total_order_num]), np.array([day_roi])
    return np.concatenate([lower_obs, median_obs, higher_obs, mean_obs, std_obs, max_obs, min_obs, day_total_order_num, day_roi], 0)
    # return np.concatenate([k_means.cluster_centers_.flatten(), mean_obs, std_obs, max_obs, min_obs, day_total_order_num, day_roi], 0)
