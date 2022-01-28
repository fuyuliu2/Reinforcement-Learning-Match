import os
import sys
# sys.path.append("")
sys.path.insert(0, os.getcwd())
import importlib
from stable_baselines3 import PPO
import virtual_env
from virtual_env import get_env_instance
import numpy as np
import pickle as pk


importlib.reload(virtual_env)

env = get_env_instance("user_states_by_day.npy", "venv.pkl")
policy = PPO.load("rl_model.zip")
validation_length = 14

total_gmv = 0.0
total_cost = 0.0
obs = env.reset()
for day_index in range(validation_length):
    coupon_action, _ = policy.predict(obs, deterministic=True)  # Some randomness will be added to action if deterministic=False
    obs, reward, done, info = env.step(coupon_action)
    if reward != 0:
        info["Reward"] = reward
    print(f"Day {day_index+1}: {info}")


with open(f"venv.pkl", "rb") as f:
    venv = pk.load(f, encoding="utf-8")

initial_states = np.load(f"user_states_by_day.npy")[10]
coupon_actions = np.array([(5, 0.95) for _ in range(initial_states.shape[0])])

node_values = venv.infer_one_step({"state": initial_states, "action_1": coupon_actions})
user_actions = node_values['action_2']
day_order_num, day_avg_fee = user_actions[..., 0].round(), user_actions[..., 1].round(2)
print(day_order_num.reshape((-1,))[:100])
print(day_avg_fee.reshape((-1,))[:100])