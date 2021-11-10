import copy
import os
import json
from ml_env_tools import MLEnv

class MLTuner():
    def __init__(self, ml_env:MLEnv, model_variant, params):
        self.params_file = os.path.join(ml_env.root_path, f"params_{model_variant}.json")
        self.search_space_file = os.path.join(ml_env.root_path, f"params_search_space_{model_variant}.json")
        self.params = params
        is_new = True
        if os.path.exists(self.search_space_file) is True:
            with open(self.search_space_file,'r') as f:
                self.search_space = json.load(f)
                print(f"Initialized search_space from {self.search_space_file}")
                is_new = False
        if is_new is True:
            self.search_space["best_params"] = copy.deepcopy(params)
            self.search_space["best_ev"] = 0
            self.search_space["is_first"] = True

    def tune(self, eval_func):
        for key in self.search_space["params"]:
            if key not in self.params:
                print(f"search_space searches for {key}, but it's not in params!")
                return self.search_space["best_params"]
            else: # initialize with first entry in search space
                self.search_space["best_params"][key]=self.search_space["params"][key][0]

        p_cnt=0
        for key in self.search_space["params"]:
            params=copy.deepcopy(self.search_space["best_params"])
            vals=self.search_space["params"][key]
            for val in vals:
                if self.search_space["is_first"] is False:
                    if val==self.search_space["best_params"][key]:
                        continue  # Was already tested.
                else:
                    self.search_space["is_first"] = False
                if p_cnt < self.search_space["progress"]:
                    p_cnt += 1
                    print(f"Fast forwarding: {key} {val}")
                    continue
                else:
                    p_cnt += 1
                self.search_space["progress"] += 1
                params[key]=val
                print(f"#### Testing: {key}={val} with {params}:")
                interrupted, ev = eval_func(params)
                print(f"] Eval: {ev}")
                if ev > self.search_space["best_ev"]:
                    self.search_space["best_ev"] = ev
                    self.search_space["best_params"] = copy.deepcopy(params)
                    print("*********************************************************")
                    print(f"Best parameter set with ev={ev}: {params}")
                    print("*********************************************************")
                    with open(self.params_file, "w") as f:
                        json.dump(self.search_space["best_params"], f, indent=4)
                    with open(self.search_space_file, "w") as f:
                        json.dump(self.search_space, f, indent=4)
                if interrupted>0:
                    return self.search_space["best_params"]
        print(f"Best parameter set with {self.search_space['best_ev']} val_loss: {self.search_space['best_params']}")
        return self.search_space["best_params"]
        