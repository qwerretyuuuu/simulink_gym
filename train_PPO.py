import os
import argparse
import string
import random
from cartpole_simulink import CartPoleSimulink  # 自定义的 Simulink 小车-倒立摆环境
from stable_baselines3 import PPO, DDPG, TD3
from pathlib import Path
from datetime import datetime
from simulink_gym import SimulinkEnv, Observation, Observations
from simulink_gym.utils import CommSocket
from stable_baselines3.common.policies import ActorCriticPolicy
import torch, gymnasium
import numpy
import time
torch.serialization.add_safe_globals([gymnasium.spaces.box.Box, numpy.dtype])

def define_parser():
    parser = argparse.ArgumentParser(
        description="使用 PPO 训练 Simulink 小车-倒立摆环境（支持 BC 权重初始化）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # 原有参数
    parser.add_argument("-a", "--gae_lambda", type=float, default=0.95)
    parser.add_argument("-b", "--batch_size", type=int, default=64)
    parser.add_argument("-B", "--benchmark", action="store_true")
    parser.add_argument("-d", "--log_dir", type=str, default="./logs")
    parser.add_argument("-e", "--num_epochs", type=int, default=10)
    parser.add_argument("-g", "--gamma", type=float, default=0.99)
    parser.add_argument("-l", "--learning_rate", type=float, default=1e-4)
    parser.add_argument("-n", "--num_steps", type=int, default=2048)
    parser.add_argument("-s", "--save_policy", action="store_true")
    parser.add_argument("-t", "--total_timesteps", type=int, default=300000)
    parser.add_argument("-v", "--verbose", type=int, default=0)
    parser.add_argument("-w", "--wandb", action="store_true")
    # 新增：预训练 BC 模型路径
    parser.add_argument("--pretrained_bc", type=str, default="models/ppo_cloned_policy.pt",
                        help="指定预训练 BC 策略路径，用于初始化 PPO Actor 权重")
    return parser

def main():
    # --- 参数解析 ---
    parser = define_parser()
    args = parser.parse_args()

    # --- 通用参数 ---
    save_policy = args.save_policy
    verbose = args.verbose
    wb = args.wandb
    benchmark = args.benchmark

    # --- PPO训练参数 ---
    total_timesteps = args.total_timesteps
    batch_size = args.batch_size
    discount_factor = args.gamma
    learning_rate = args.learning_rate
    num_steps = args.num_steps
    num_epochs = args.num_epochs
    gae_lambda = args.gae_lambda

    # --- 构造日志路径 ---
    timestamp = datetime.now().strftime("%Y%m%d.%H%M%S")
    random_tag = "".join(random.choices(string.ascii_lowercase + string.digits, k=8))
    run_id = f"{timestamp}-{random_tag}"
    log_dir = Path(args.log_dir).resolve().joinpath(run_id)
    log_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "total_timesteps": total_timesteps,
        "batch_size": batch_size,
        "discount_factor": discount_factor,
        "learning_rate": learning_rate,
        "num_steps": num_steps,
        "num_epochs": num_epochs,
        "gae_lambda": gae_lambda,
        # TD3 专属参数（保留）
        "buffer_size": 1_000_000,
        "learning_starts": 100,
        "tau": 0.005,
        "train_freq": 1,
        "gradient_steps": 1,
        "action_noise": None,
        "policy_delay": 3,
        "target_policy_noise": 0.2,
        "target_noise_clip": 0.5,
        "seed": None,
    }

    # --- Weights & Biases 日志记录 ---
    if wb:
        import wandb
        from wandb.integration.sb3 import WandbCallback
        os.environ["WANDB_DISABLE_GIT"] = "True"
        run = wandb.init(
            project="simulink_gym",
            group="simulink_cartpole_env" if not benchmark else "gym_cartpole_env",
            job_type="examples",
            tags=["PPO"],
            sync_tensorboard=True,
            config=config,
            dir=log_dir,
            save_code=False,
            id=run_id,
        )
        callback = WandbCallback()
    else:
        callback = None

    # --- 创建环境 ---
    if not benchmark:
        env = CartPoleSimulink()
        print("创建环境\n")
    else:
        import gym
        env = gym.make("CartPole-v1")

    # --- 创建 PPO agent ---
    agent = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=dict(net_arch=[256, 256]),
        batch_size=config["batch_size"],
        gamma=config["discount_factor"],
        learning_rate=config["learning_rate"],
        n_steps=config["num_steps"],
        n_epochs=config["num_epochs"],
        gae_lambda=config["gae_lambda"],
        verbose=verbose,
        tensorboard_log=str(log_dir),
    )


    if args.pretrained_bc is not None and Path(args.pretrained_bc).exists():
        print(f"正在加载预训练策略: {args.pretrained_bc}")
        agent.policy.load_state_dict(torch.load("models/ppo_cloned_policy.pt"), strict=False)

        print("✅ 成功用 BC 预训练权重初始化 PPO Actor")

    # --- 开始训练 ---
    agent.learn(
        total_timesteps=config["total_timesteps"],
        log_interval=4,
        callback=callback,
        progress_bar=True,
    )

    # --- 训练结束保存策略 ---
    if save_policy:
        policy = agent.policy
        policy.save(f"{log_dir}/learned_policy")

    env.close()
    if wb:
        run.finish()

if __name__ == "__main__":
    main()
