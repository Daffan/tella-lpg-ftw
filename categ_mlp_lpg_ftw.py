import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical, kl_divergence


class MLPLPGFTW:
    def __init__(self, observation_space, action_space,
                 k=3,
                 hidden_size=64,
                 seed=None,
                 max_k=None,
                 use_gpu=False):
        """
        :param observation_space: from gym environment
        :param action_space: from gym environment
        :param hidden_size: output dimension of convnet (one more task-specific layer is added)
        :param seed: random seed
        :param feat_cutoff_layer: layer of resnet to cut features
        """
        self.obs_dims = np.prod(observation_space.shape)
        self.m = action_space.n  # number of actions
        self.k = k      # number of elements in the dictionary (LPG-FTW)
        self.max_k = max_k

        self.device = 'cuda' if use_gpu else 'cpu'

        # Set seed
        # ------------------------
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Policy network
        # ------------------------
        self.model = MLPMuNet(self.obs_dims, self.m, hidden_size, self.k, self.max_k, device=self.device).to(self.device)
        self.tasks_seen = []

        # Old Policy network
        # ------------------------
        self.old_model = MLPMuNet(self.obs_dims, self.m, hidden_size, self.k, self.max_k, device=self.device).to(self.device)
        self.old_model.L.data = self.model.L.data.clone()

        # Placeholders
        # ------------------------
        self.obs_var = Variable(torch.randn(self.obs_dims), requires_grad=False).to(self.device)

    def set_task(self, task_id):
        self.task_id = task_id
        first_time = False
        if self.task_id not in self.tasks_seen:
            first_time = True
            self.tasks_seen.append(task_id)

        self.model.set_task(task_id)
        self.old_model.set_task(task_id)
        self.old_model.S[task_id] = Variable(self.model.S[task_id]).to(self.device)
        self.old_model.epsilon_col = Variable(self.model.epsilon_col).to(self.device)

        if first_time:
            self.old_model.fc_out[task_id].load_state_dict(self.model.fc_out[task_id].state_dict())
            self.old_model.L.data = self.model.L.data.clone()

        if self.model.T <= self.k:
            self.trainable_params = [self.model.L]
            self.old_params = [self.old_model.L]
        elif self.model.T <= self.max_k:
            print("Training L and S")
            self.trainable_params = [self.model.S[task_id], self.model.epsilon_col]
            self.old_params = [self.old_model.S[task_id], self.old_model.epsilon_col]
        else:       # if self.model.T > self.k
            print("Training only S")
            self.trainable_params = [self.model.S[task_id]]
            self.old_params = [self.old_model.S[task_id]]

        self.trainable_params += list(self.model.fc_out[task_id].parameters())
        self.old_params += list(self.old_model.fc_out[task_id].parameters())
        self.param_shapes = [p.data.cpu().numpy().shape for p in self.trainable_params]
        self.param_sizes = [p.data.cpu().numpy().size for p in self.trainable_params]
        self.d = np.sum(self.param_sizes)

    # Utility functions
    # ============================================
    def get_param_values(self, task_id):
        print(self.model.S[task_id])
        params = np.concatenate([p.contiguous().view(-1).data.cpu().numpy()
                                 for p in self.trainable_params])
        return params.copy()

    def set_param_values(self, new_params, task_id, set_new=True, set_old=True):
        if set_new:
            current_idx = 0
            for idx, param in enumerate(self.trainable_params):
                vals = new_params[current_idx:current_idx + self.param_sizes[idx]]
                vals = vals.reshape(self.param_shapes[idx])
                param.data = torch.from_numpy(vals).float().to(self.device)
                current_idx += self.param_sizes[idx]
        if set_old:
            current_idx = 0
            for idx, param in enumerate(self.old_params):
                vals = new_params[current_idx:current_idx + self.param_sizes[idx]]
                vals = vals.reshape(self.param_shapes[idx])
                param.data = torch.from_numpy(vals).float().to(self.device)
                current_idx += self.param_sizes[idx]

    # Main functions
    # ============================================
    def get_action(self, observation):
        o = np.float32(observation)
        self.obs_var.data = torch.from_numpy(o).to(self.device)
        dist = Categorical(logits=self.model(self.obs_var))
        """print(self.task_id)
        if self.task_id >= 2:
            print("Logits")
            print(dist.logits.data.cpu().numpy())
            print("Probs:")
            print(dist.probs.data.cpu().numpy())"""
        return [dist.sample().data.cpu().numpy(), {'log_probs': dist.logits.data.cpu().numpy(),
                                                   'evaluation': dist.probs.data.cpu().numpy().argmax(axis=1)}]

    def get_dist(self, observations, model=None):
        model = self.model if model is None else model
        obs_var = Variable(torch.from_numpy(observations).float(), requires_grad=False).to(self.device)
        return Categorical(logits=model(obs_var))

    def log_likelihood(self, observations, actions, model=None):
        dist = self.get_dist(observations, model)
        act_var = Variable(torch.from_numpy(actions).long(), requires_grad=False).to(self.device)
        return dist.log_prob(act_var).data.cpu().numpy()

    def _dist_info(self, observations, actions, model):
        dist = self.get_dist(observations, model=model)
        act_var = Variable(torch.from_numpy(actions).long(), requires_grad=False).to(self.device)
        LL = dist.log_prob(act_var)
        return LL, dist

    def old_dist_info(self, observations, actions):
        return self._dist_info(observations, actions, self.old_model)

    def new_dist_info(self, observations, actions):
        return self._dist_info(observations, actions, self.model)

    def likelihood_ratio(self, new_dist_info, old_dist_info):
        LL_old = old_dist_info[0]
        LL_new = new_dist_info[0]
        LR = torch.exp(LL_new - LL_old)
        if LR.mean() > 1000:
            print(self.model.L, self.old_model.L)
            print(self.model.S[self.task_id], self.old_model.S[self.task_id])
            print(self.log_std[self.task_id], self.old_log_std[self.task_id])
        return LR

    def mean_kl(self, new_dist_info, old_dist_info):
        sample_kl = kl_divergence(old_dist_info[1], new_dist_info[1])
        return torch.mean(sample_kl)


class BasicMLP(nn.Module):
    def __init__(self, in_dim, hidden_size, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_size = hidden_size
        self.out_dim = out_dim

        self.net = nn.Sequential()
        self.net.add_module('fc1', nn.Linear(in_dim, hidden_size))
        self.net.add_module('relu1', nn.ReLU())
        self.net.add_module('fc2', nn.Linear(hidden_size, out_dim))
        self.net.add_module('relu2', nn.ReLU())

    def forward(self, inputs):
        return self.net(inputs)

    @classmethod
    def forward_from_column(cls, column, inp, in_dim, hidden_size, out_dim):
        """
        Make sure that any changes in network's construction (in __init__) are
        reflected here
        """
        i_0 = 0
        # layer 1
        i_f = i_0 + (in_dim * hidden_size)
        weight = column[i_0:i_f].view(hidden_size, in_dim)
        i_0 = i_f
        i_f = i_0 + hidden_size
        bias = column[i_0:i_f].view(-1)
        i_0 = i_f
        out = F.linear(inp, weight, bias)
        out = F.relu(out)
        # layer 2
        i_f = i_0 + (hidden_size * out_dim)
        weight = column[i_0:i_f].view(out_dim, hidden_size)
        i_0 = i_f
        i_f = i_0 + out_dim
        bias = column[i_0:i_f].view(-1)
        i_0 = i_f
        out = F.linear(out, weight, bias)
        out = F.relu(out)

        assert i_f == column.view(-1).shape[0], "Something went wrong. Did not use all parameters in column"
        return out

    def to_column(self):
        """
        Reshape parameters into single column
        """
        flat_params = []
        for module in self.net.modules():
            if hasattr(module, 'weight'):
                flat_params.append(module.weight.contiguous().view(-1).data)
            if hasattr(module, 'bias'):
                flat_params.append(module.bias.contiguous().view(-1).data)
        return torch.cat(flat_params)



class MLPMuNet(nn.Module):
    def __init__(self, obs_dims, act_dim, hidden_size, dict_dim,
                 max_dict_dim=None,
                 device='cpu'):
        super(MLPMuNet, self).__init__()

        self.obs_dims = obs_dims
        self.act_dim = act_dim
        self.dict_dim = dict_dim
        self.max_dict_dim = max_dict_dim
        self.T = 0
        self.hidden_size = hidden_size
        self.device = device

        # self.feature_extractor = ResNet18Features(cutoff_layer=cutoff_layer).to(self.device)
        # self.feature_extractor.requires_grad_(False)
        columns = []
        for k in range(self.dict_dim):
            net = BasicMLP(self.obs_dims, hidden_size, hidden_size)

            columns.append(net.to_column().to(self.device))

        self.L = torch.stack(columns, dim=1)
        self.L.requires_grad = True
        self.S = {}
        self.fc_out = {}
        self.use_theta = False
    
    def set_use_theta(self, use, add_epsilon=False):
        self.use_theta = use
        if use:
            if add_epsilon:
                self.theta = torch.autograd.Variable(torch.mm(self.L, self.S[self.task_id]) + self.epsilon_col, requires_grad=True).to(self.device)
            else:
                self.theta = torch.autograd.Variable(torch.mm(self.L, self.S[self.task_id]), requires_grad=True).to(self.device)
        else:
            self.theta = torch.autograd.Variable(torch.zeros(0)).to(self.device)

    def set_task(self, task_id):
        self.task_id = task_id

        if task_id not in self.S:
            self.fc_out[task_id] = nn.Linear(self.hidden_size, self.act_dim).to(self.device)
            for param in self.fc_out[task_id].parameters():
                param.data = param.data * 1e-2
            if self.T < self.dict_dim:
                s = np.zeros((self.dict_dim, 1))
                s[self.T] = 1
                self.S[task_id] = Variable(torch.from_numpy(s).float(), requires_grad=True).to(self.device)
                self.epsilon_col = torch.zeros(self.L.shape[0], 1, requires_grad=False).to(self.device)
            elif self.dict_dim < self.max_dict_dim:
                self.S[task_id] = Variable(torch.stack(list(self.S.values())).mean(0), requires_grad=True).to(self.device)    # mean of previous values
                net = BasicMLP(self.obs_dims, self.hidden_size, self.hidden_size)
                self.epsilon_col = net.to_column().to(self.device).view(-1, 1)

                assert self.epsilon_col.shape[0] == self.L.shape[0]
                self.epsilon_col.requires_grad = True
            else:
                self.S[task_id] = Variable(torch.stack(list(self.S.values())).mean(0), requires_grad=True).to(self.device)    # mean of previous values
                self.epsilon_col = torch.zeros(self.L.shape[0], 1, requires_grad=False).to(self.device)
            self.T += 1

    def forward(self, x):
        out = x  # out = (x - self.in_shift) / (self.in_scale + 1e-8)
        if not self.use_theta:
            theta = torch.mm(self.L, self.S[self.task_id]) + self.epsilon_col

        out = BasicMLP.forward_from_column(theta, out, self.obs_dims, self.hidden_size, self.hidden_size)
        out = self.fc_out[self.task_id](out)

        # out = out * self.out_scale + self.out_shift
        return out
