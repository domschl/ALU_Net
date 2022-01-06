import copy
import os
import json
from ml_env_tools import MLEnv

class MLTuner():
    def __init__(self, ml_env:MLEnv, model_variant, load_progress=False):
        self.search_space_file = os.path.join(ml_env.project_path, f"params_search_space_{model_variant}.json")
        is_new = True
        if os.path.exists(self.search_space_file) is True and load_progress is True:
            with open(self.search_space_file,'r') as f:
                self.search_space = json.load(f)
                print(f"Initialized search_space from {self.search_space_file}")
                is_new = False
        if is_new is True:
            self.search_space = {}
            self.search_space["best_ev"] = 0
            self.search_space["is_first"] = True
            self.search_space["progress"] = 0

    def tune(self, param_space, eval_func):
        if "best_params" not in self.search_space:
            self.search_space["best_params"]={}
        for key in param_space:
            if key not in self.search_space["best_params"]:
                self.search_space["best_params"][key]=param_space[key][0]
        p_cnt=0
        for key in param_space:
            params=copy.deepcopy(self.search_space["best_params"])
            vals=param_space[key]
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
                ev = eval_func(params)
                print(f"] Eval: {ev}")
                if ev > self.search_space["best_ev"]:
                    self.search_space["best_ev"] = ev
                    self.search_space["best_params"] = copy.deepcopy(params)
                    print("*********************************************************")
                    print(f"Best parameter set with ev={ev}: {params} -> {self.search_space_file}")
                    print("*********************************************************")
                    with open(self.search_space_file, "w") as f:
                        json.dump(self.search_space, f, indent=4)
        print(f"Best parameter set with {self.search_space['best_ev']} eval: {self.search_space['best_params']}")
        return self.search_space["best_params"]
        