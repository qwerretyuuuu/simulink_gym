import argparse
import pathlib
from datetime import datetime
import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
# ✨ 核心修改：我们现在使用 PPO ✨
from stable_baselines3 import PPO 
from stable_baselines3.common.logger import configure
from imitation.algorithms import bc
from imitation.data import types

class DummyEnv(gym.Env):
    """一个最小化的模拟Gymnasium环境，用于在没有真实环境的情况下初始化SB3模型。"""
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space
    def step(self, action): raise NotImplementedError
    def reset(self, seed=None, options=None): raise NotImplementedError

def main(args):
    # --- 1 & 2. 加载数据并划分 ---
    print(f"正在从 '{args.state_file}' 和 '{args.action_file}' 加载专家数据...")
    try:
        df_states = pd.read_excel(args.state_file, header=None)
        df_actions = pd.read_excel(args.action_file, header=None)
    except FileNotFoundError as e:
        print(f"错误: 无法找到数据文件！"); print(e); return
    expert_obs = df_states.to_numpy().astype('float32')
    expert_actions = df_actions.iloc[:, :4].to_numpy().astype('float32')
    if expert_obs.shape[0] != expert_actions.shape[0]:
        print(f"错误：样本数量不匹配！"); return

    print(f"数据加载成功！共 {expert_obs.shape[0]} 条样本。正在划分为训练集和验证集...")
    train_obs, val_obs, train_acts, val_acts = train_test_split(
        expert_obs, expert_actions, test_size=args.val_split, random_state=args.seed
    )
    print(f"训练集大小: {len(train_obs)} | 验证集大小: {len(val_obs)}")
    train_demonstrations = types.Transitions(
        obs=train_obs, acts=train_acts, infos=np.array([{} for _ in range(len(train_obs))]),
        dones=np.zeros(len(train_obs), dtype=bool), next_obs=train_obs.copy(),
    )
    
    # --- 3. 定义空间 ---
    obs_dim = expert_obs.shape[1]
    action_dim = expert_actions.shape[1]
    observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
    action_space = gym.spaces.Box(low=-1, high=1, shape=(action_dim,), dtype=np.float32)

    # --- 4. 设置日志 ---
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = pathlib.Path(args.log_dir) / f"bc_ppo_seed{args.seed}_{timestamp}"
    print(f"TensorBoard 日志将被保存在: {log_dir}")
    custom_logger = configure(str(log_dir), ["stdout", "tensorboard"])
    
    # --- 5. 初始化 BC 训练器 ---
    print("正在初始化 BC 训练器...")
    dummy_env = DummyEnv(observation_space, action_space)
    
    # 使用 PPO 策略，因为它有 BC 训练器需要的 .evaluate_actions() 方法
    temp_model = PPO(
        policy="MlpPolicy",
        env=dummy_env,
        policy_kwargs=dict(net_arch=args.net_arch),
    )
    policy = temp_model.policy
    rng = np.random.default_rng(args.seed)

    bc_trainer = bc.BC(
        observation_space=observation_space,
        action_space=action_space,
        policy=policy,
        demonstrations=train_demonstrations,
        rng=rng,
        batch_size=args.batch_size,
        optimizer_cls=torch.optim.Adam,
        optimizer_kwargs=dict(lr=args.lr),
        custom_logger=custom_logger,
    )
    
    # --- 6. 训练 ---
    print("\n--- 开始行为克隆训练 ---")
    print(f"要查看训练过程，请在新的终端运行: tensorboard --logdir {args.log_dir}")
    bc_trainer.train(n_epochs=args.epochs, progress_bar=True)
    print("--- 训练完成！ ---\n")

    # --- 7. 评估 ---
    print("--- 正在验证集上评估最终模型性能 ---")
    val_obs_tensor = torch.as_tensor(val_obs, device=bc_trainer.policy.device)
    val_acts_tensor = torch.as_tensor(val_acts, device=bc_trainer.policy.device)
    trained_policy = bc_trainer.policy
    trained_policy.eval()
    with torch.no_grad():
        # 对于 ActorCriticPolicy (来自PPO), 动作直接由 policy 对象本身预测
        # 而不是 policy.actor
        pred_acts_tensor, _, _ = trained_policy(val_obs_tensor)
    mse_loss_fn = torch.nn.MSELoss()
    validation_loss = mse_loss_fn(pred_acts_tensor, val_acts_tensor)
    print(f"最终验证集上的均方误差 (MSE Loss): {validation_loss.item():.6f}")

    # --- 8. 保存 ---
    save_path = pathlib.Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(trained_policy.state_dict(),args.save_path)
    # trained_policy.save(save_path)
    print(f"克隆策略已成功保存到: {save_path}")


def create_parser():
    parser = argparse.ArgumentParser(description="使用专家数据进行行为克隆训练的脚本")
    parser.add_argument("--state-file", type=str, default="state_AI_VDC.xlsx")
    parser.add_argument("--action-file", type=str, default="action_AI_VDC.xlsx")
    # ✨ 修改了默认文件名
    parser.add_argument("--save-path", type=str, default="models/ppo_cloned_policy.pt")
    parser.add_argument("--log-dir", type=str, default="logs/")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--net-arch", nargs='+', type=int, default=[256, 256])
    parser.add_argument("--seed", type=int, default=42)
    return parser

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
