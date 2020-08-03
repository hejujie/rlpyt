import sys

from rlpyt.agents.dqn.atari.atari_dqn_agent import AtariDqnAgent
from rlpyt.algos.dqn.dqn import DQN
from rlpyt.envs.atari.atari_env import AtariEnv
from rlpyt.envs.atari.atari_env import AtariTrajInfo
from rlpyt.experiments.configs.atari.dqn.atari_dqn import configs
from rlpyt.runners.minibatch_rl import MinibatchRl
from rlpyt.samplers.paralell.gpu.collectors import GpuWaitResetCollector
from rlpyt.samplers.parallel.gpu.sampler import GpuSampler
from rlpyt.utils.launching.affinity import affinity_from_code
from rlpyt.utils.launching.variant import load_variant
from rlpyt.utils.launching.variant import update_config
from rlpyt.utils.logging.context import logger_context


def build_and_train(slot_affinity_code, log_dir, run_ID, config_key):
    affinity = affinity_from_code(slot_affinity_code)
    config = configs[config_key]
    variant = load_variant(log_dir)
    config = update_config(config, variant)
    config["eval_env"]["game"] = config["env"]["game"]

    sampler = GpuSampler(
        EnvCls=AtariEnv,
        env_kwargs=config["env"],
        CollectorCls=GpuWaitResetCollector,
        TrajInfoCls=AtariTrajInfo,
        eval_env_kwargs=config["eval_env"],
        **config["sampler"]
    )
    algo = DQN(optim_kwargs=config["optim"], **config["algo"])
    agent = AtariDqnAgent(model_kwargs=config["model"], **config["agent"])
    runner = MinibatchRl(
        algo=algo, agent=agent, sampler=sampler, affinity=affinity, **config["runner"]
    )
    name = config["env"]["game"]
    with logger_context(log_dir, run_ID, name, config):
        runner.train()


if __name__ == "__main__":
    build_and_train(*sys.argv[1:])
