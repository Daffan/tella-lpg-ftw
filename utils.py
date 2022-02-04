import matplotlib
matplotlib.use('Agg')
import numpy as np
import pickle
import csv


# Copied from mjrl.utils.process_samples.py
def compute_returns(paths, gamma):
    for path in paths:
        path["returns"] = discount_sum(path["rewards"], gamma)

def compute_advantages(paths, baseline, gamma, gae_lambda=None, normalize=False):
    # compute and store returns, advantages, and baseline 
    # standard mode
    if gae_lambda == None or gae_lambda < 0.0 or gae_lambda > 1.0:
        for path in paths:
            path["baseline"] = baseline.predict(path)
            path["advantages"] = path["returns"] - path["baseline"]
        if normalize:
            alladv = np.concatenate([path["advantages"] for path in paths])
            mean_adv = alladv.mean()
            std_adv = alladv.std()
            for path in paths:
                path["advantages"] = (path["advantages"]-mean_adv)/(std_adv+1e-8)
    # GAE mode
    else:
        for path in paths:
            b = path["baseline"] = baseline.predict(path)
            if b.ndim == 1:
                b1 = np.append(path["baseline"], 0.0 if path["terminated"] else b[-1])
            else:
                b1 = np.vstack((b, np.zeros(b.shape[1]) if path["terminated"] else b[-1]))
            td_deltas = path["rewards"] + gamma*b1[1:] - b1[:-1]
            path["advantages"] = discount_sum(td_deltas, gamma*gae_lambda)
        if normalize:
            alladv = np.concatenate([path["advantages"] for path in paths])
            mean_adv = alladv.mean()
            std_adv = alladv.std()
            for path in paths:
                path["advantages"] = (path["advantages"]-mean_adv)/(std_adv+1e-8)

def compute_LL(paths, policy):
    for path in paths:
        path["LL"] = policy.new_dist_info(path["observations"], path["actions"])[0]

def discount_sum(x, gamma, terminal=0.0):
    y = []
    run_sum = terminal
    for t in range( len(x)-1, -1, -1):
        run_sum = x[t] + gamma*run_sum
        y.append(run_sum)

    return np.array(y[::-1])


# Copied from mjrl.utils.cg_solver
def cg_solve(f_Ax, b, x_0=None, cg_iters=1, residual_tol=1e-10):
    x = np.zeros_like(b) #if x_0 is None else x_0
    r = b.copy() #if x_0 is None else b-f_Ax(x_0)
    p = r.copy()
    rdotr = r.dot(r)

    for i in range(cg_iters):
        z = f_Ax(p)
        v = rdotr / p.dot(z)
        #print(v)
        x += v * p
        #print(np.linalg.norm(x))
        r -= v * z
        newrdotr = r.dot(r)
        mu = newrdotr / rdotr
        p = r + mu * p

        rdotr = newrdotr
        if rdotr < residual_tol:
            break
    #print("Done HVP")

    return x


# Copied from mjrl.utils.logger
class DataLog:

    def __init__(self):
        self.log = {}
        self.max_len = 0

    def log_kv(self, key, value):
        # logs the (key, value) pair
        if key not in self.log:
            self.log[key] = []
        self.log[key].append(value)
        if len(self.log[key]) > self.max_len:
            self.max_len = self.max_len + 1

    def save_log(self, save_path):
        pickle.dump(self.log, open(save_path+'/log.pickle', 'wb'))
        with open(save_path+'/log.csv', 'w') as csv_file:
            fieldnames = self.log.keys()
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for row in range(self.max_len):
                row_dict = {}
                for key in self.log.keys():
                    if row < len(self.log[key]):
                        row_dict[key] = self.log[key][row]
                writer.writerow(row_dict)

    def get_current_log(self):
        row_dict = {}
        for key in self.log.keys():
            row_dict[key] = self.log[key][-1]
        return row_dict

    def read_log(self, log_path):
        with open(log_path) as csv_file:
            reader = csv.DictReader(csv_file)
            listr = list(reader)
            keys = reader.fieldnames
            data = {}
            for key in keys:
                data[key] = []
            for row in listr:
                for key in keys:
                    try:
                        data[key].append(eval(row[key]))
                    except:
                        None
        self.log = data


# Copied from mjrl.utils.process_samples
def compute_returns(paths, gamma):
    for path in paths:
        path["returns"] = discount_sum(path["rewards"], gamma)

def compute_advantages(paths, baseline, gamma, gae_lambda=None, normalize=False):
    # compute and store returns, advantages, and baseline 
    # standard mode
    if gae_lambda == None or gae_lambda < 0.0 or gae_lambda > 1.0:
        for path in paths:
            path["baseline"] = baseline.predict(path)
            path["advantages"] = path["returns"] - path["baseline"]
        if normalize:
            alladv = np.concatenate([path["advantages"] for path in paths])
            mean_adv = alladv.mean()
            std_adv = alladv.std()
            for path in paths:
                path["advantages"] = (path["advantages"]-mean_adv)/(std_adv+1e-8)
    # GAE mode
    else:
        for path in paths:
            b = path["baseline"] = baseline.predict(path)
            if b.ndim == 1:
                b1 = np.append(path["baseline"], 0.0 if path["terminated"] else b[-1])
            else:
                b1 = np.vstack((b, np.zeros(b.shape[1]) if path["terminated"] else b[-1]))
            td_deltas = path["rewards"] + gamma*b1[1:] - b1[:-1]
            path["advantages"] = discount_sum(td_deltas, gamma*gae_lambda)
        if normalize:
            alladv = np.concatenate([path["advantages"] for path in paths])
            mean_adv = alladv.mean()
            std_adv = alladv.std()
            for path in paths:
                path["advantages"] = (path["advantages"]-mean_adv)/(std_adv+1e-8)

def compute_LL(paths, policy):
    for path in paths:
        path["LL"] = policy.new_dist_info(path["observations"], path["actions"])[0]

def discount_sum(x, gamma, terminal=0.0):
    y = []
    run_sum = terminal
    for t in range( len(x)-1, -1, -1):
        run_sum = x[t] + gamma*run_sum
        y.append(run_sum)

    return np.array(y[::-1])