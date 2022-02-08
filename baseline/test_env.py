import json
import json
import numpy as np

# Use user states from first day (day 31) as initial states
import pandas as pd

initial_states = np.load(f"user_states_by_day.npy")[4]

# Send no coupon (0, 1.00) to all users
zero_actions = np.array([(0, 1) for _ in range(initial_states.shape[0])])

#%% md

# Then users' response action could be predicted by virtual environment in this way:

#%%

import pickle as pk

with open(f"venv.pkl", "rb") as f:
    venv = pk.load(f, encoding="utf-8")

# Propogate from initial_states and zero_actions for one step, returning all nodes' values after propogated
node_values = venv.infer_one_step({ "state": initial_states, "action_1": zero_actions })
user_action = node_values["action_2"]
pd.DataFrame(node_values["state"]).to_csv('state.csv')
pd.DataFrame(node_values["next_state"]).to_csv('next_state.csv')
pd.DataFrame(node_values["action_1"]).to_csv('coupon_action.csv')
pd.DataFrame(user_action).to_csv('user_action4.csv')


# print("Node values after propogated for one step:", node_values)
# print("Predicted user actions:", user_action)
