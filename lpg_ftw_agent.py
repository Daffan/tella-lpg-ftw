from copy import deepcopy
import typing
import logging

import gym
import tella

from categ_mlp_lpg_ftw import MLPLPGFTW
from npg_cg_ftw import NPGFTW
from mlp_baselines import MLPBaseline

logger = logging.getLogger("LPG_FTW Agent")
DEVICE = "cpu"

# Constants copied from experiments.habitat_ste_m15.py
BASELINE_TRAINING_EPOCH = 20
NORMALIZED_STEP_SIZE = 0.00001
HVP_SAMPLEFRAC = 0.00833333333
BATCH_SIZE = 128

N = 50
GAMMA = 0.995
GAE_LAMBDA = None # 0.97

POLICY_HIDDEN_SIZE = 128
BASELINE_HIDDEN_SIZE = 128
K=1
MAX_K=2

BASELINE_LR = 1e-5

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
        
        self.agent_train = NPGFTW(
            policy,
            baselines,
            num_envs=num_envs,
            normalized_step_size=NORMALIZED_STEP_SIZE,
            seed=rng_seed,
            use_gpu=(DEVICE != "cpu"),
            hvp_sample_frac=HVP_SAMPLEFRAC,
            batch_size=BATCH_SIZE
        )
        self.agent = self.agent_train
        
        self.train = None # True for learning_block and False for evaluation_block
    
    def block_start(self, is_learning_allowed: bool) -> None:
        super().block_start(is_learning_allowed)
        if is_learning_allowed:
            logger.info("About to start a new learning block")
            self.training = True
            self.agent = self.agent_train
        else:
            logger.info("About to start a new evaluation block")
            self.training = False
            
    def task_start(self, task_name: typing.Optional[str]) -> None:
        logger.info(
            f"\tAbout to start interacting with a new task. task_name={task_name}"
        )
        if not task_name in self.agent.baselines.keys():
            self.agent.baselines[task_name] = MLPBaseline(
                self.observation_space,
                reg_coef=1e-3,
                batch_size=BASELINE_HIDDEN_SIZE,
                epochs=BASELINE_TRAINING_EPOCH,
                learn_rate=BASELINE_LR,
                use_gpu=(DEVICE != "cpu")
            )
        self.agent.set_task(task_name)
        self.agent.rollout_buffer.clear_buffer()
        
    def choose_actions(
        self, observations: typing.List[typing.Optional[tella.Observation]]
    ) -> typing.List[typing.Optional[tella.Action]]:
        # Don't know whether torch.no_grad is needed or not
        # In original code, they didn't use torch._no_grad for eval
        actions = []
        for obs in observations:
            if obs is None:
                actions.append(None)
            else:
                obs = obs.reshape(1, -1)
                a, agent_info = self.agent.policy.get_action(obs)
                if self.training:
                    actions.append(a[0]) # sampled from distribution
                else:
                    actions.append(agent_info['evaluation'][0]) # maximum likelihood
        return actions

    def receive_transitions(self, transitions: typing.List[typing.Optional[tella.Transition]]) -> None:
        assert len(transitions) == self.num_envs
        if not self.is_learning_allowed:
            return
        
        transitions = [self.flat_observation(t) for t in transitions]
        
        self.agent.train_step(
            transitions,
            N=N, gamma=GAMMA, gae_lambda=GAE_LAMBDA
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