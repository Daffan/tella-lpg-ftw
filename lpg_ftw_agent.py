from copy import deepcopy
import typing
import logging

import numpy as np
import gym
import tella

from categ_mlp_lpg_ftw import MLPLPGFTW
from npg_cg_ftw import NPGFTW
from mlp_baselines import MLPBaseline

logger = logging.getLogger("LPG_FTW Agent")
DEVICE = "cuda:0"

# Constants copied from experiments.habitat_ste_m15.py
BASELINE_TRAINING_EPOCH = 10
NORMALIZED_STEP_SIZE = 0.001
HVP_SAMPLEFRAC = 0.02
BATCH_SIZE = 128

N = 50
GAMMA = 0.995
GAE_LAMBDA = 0.97

POLICY_HIDDEN_SIZE = 128
BASELINE_HIDDEN_SIZE = 128
K=1
MAX_K=4

BASELINE_LR = 1e-6

class LpgFtwAgent(tella.ContinualRLAgent):
    def __init__(
        self,
        rng_seed: int,
        observation_space: gym.Space,
        action_space: gym.Space,
        num_envs: int,
        config_file: typing.Optional[str] = None,
    ) -> None:
        rng_seed = rng_seed % 2**32  # exceed the range limit of numpy seeding
        super(LpgFtwAgent, self).__init__(
            rng_seed, observation_space, action_space, num_envs, config_file
        )
        
        baselines = {}
        
        policy = MLPLPGFTW(
            observation_space,
            action_space,
            hidden_size=POLICY_HIDDEN_SIZE,
            k=K, max_k=MAX_K,
            seed=rng_seed,
            use_gpu=(DEVICE != "cpu")
        )
        
        self.agent = NPGFTW(
            policy,
            baselines,
            num_envs=num_envs,
            normalized_step_size=NORMALIZED_STEP_SIZE,
            seed=rng_seed,
            hvp_sample_frac=HVP_SAMPLEFRAC,
            use_gpu=(DEVICE != "cpu")
        )
        
        self.train = None # True for learning_block and False for evaluation_block
        self.use_random_policy = False
    
    def block_start(self, is_learning_allowed: bool) -> None:
        super().block_start(is_learning_allowed)
        if is_learning_allowed:
            logger.info("About to start a new learning block")
            self.training = True
        else:
            logger.info("About to start a new evaluation block")
            self.training = False
            
    def task_start(self, task_name: typing.Optional[str]) -> None:
        logger.info(
            f"\tAbout to start interacting with a new task. task_name={task_name}"
        )
        seen_tasks = list(self.agent.all_baseline.keys())
        self.task_name = task_name
        if len(seen_tasks) == 0 and not self.training: # first eval block without any training
            self.use_random_policy = True
        elif not task_name in seen_tasks:
            self.use_random_policy = False
            if self.training:
                self.agent.all_baseline[task_name] = MLPBaseline(
                    self.observation_space,
                    reg_coef=1e-3,
                    batch_size=BATCH_SIZE,
                    epochs=BASELINE_TRAINING_EPOCH,
                    learn_rate=BASELINE_LR,
                    use_gpu=(DEVICE != "cpu")
                )
                self.agent.set_task(task_name)
                self.agent.rollout_buffer.clear_log()
            else:
                self.agent.set_task(seen_tasks[-1]) # use the policy of last trained task if the task has not been trained
                self.agent.rollout_buffer.clear_log()
        else:
            self.use_random_policy = False
            self.agent.set_task(task_name)
            self.agent.rollout_buffer.clear_log()
            
        
    def choose_actions(
        self, observations: typing.List[typing.Optional[tella.Observation]]
    ) -> typing.List[typing.Optional[tella.Action]]:
        # Don't know whether torch.no_grad is needed or not
        # In original code, they didn't use torch._no_grad for eval
        if self.use_random_policy:
            actions = [None if obs is None else self.action_space.sample() for obs in observations]
        else:
            actions = []
            obs_new = [obs if obs is not None else self.observation_space.sample() for obs in observations]
            obs_new = np.stack(obs_new)
            obs_new = obs_new.reshape(obs_new.shape[0], -1)
            acts, act_infos = self.agent.policy.get_action(obs_new)
            actions = []
            for a, ai, obs in zip(acts, act_infos['evaluation'], observations):
                if obs is None:
                    actions.append(None)
                else:
                    if self.training:
                        actions.append(a)
                    else:
                        actions.append(ai)
            
        return actions

    def receive_transitions(self, transitions: typing.List[typing.Optional[tella.Transition]]) -> None:
        assert len(transitions) == self.num_envs
        if not self.is_learning_allowed:
            return
        
        transitions = [self.flat_observation(t) for t in transitions]
        
        self.agent.train_step(
            N,
            transitions,
            gamma=GAMMA,
            gae_lambda=GAE_LAMBDA,
            task_id=self.task_name
        )

    def flat_observation(self, transition: typing.Optional[tella.Transition]):
        if transition is None:
            return transition
        else:
            s, a, r, d, ns = transition
            s = s.reshape(-1)
            return s, a, r, d, ns

    def task_variant_start(
        self,
        task_name: typing.Optional[str],
        variant_name: typing.Optional[str],
    ) -> None:
        logger.info(
            f"\tAbout to start interacting with a new task variant. "
            f"task_name={task_name} variant_name={variant_name}"
        )

    def task_end(
        self,
        task_name: typing.Optional[str],
    ) -> None:
        logger.info(f"\tDone interacting with task. task_name={task_name}")

    def task_variant_end(
        self,
        task_name: typing.Optional[str],
        variant_name: typing.Optional[str],
    ) -> None:
        logger.info(
            f"\tDone interacting with task variant. "
            f"task_name={task_name} variant_name={variant_name}"
        )

    def block_end(self, is_learning_allowed: bool) -> None:
        if is_learning_allowed:
            logger.info("Done with learning block")
        else:
            logger.info("Done with evaluation block")

        
if __name__ == "__main__": 
    logging.basicConfig(level=logging.INFO)
    tella.rl_cli(LpgFtwAgent)
