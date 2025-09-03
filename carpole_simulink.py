import sys
from simulink_gym import SimulinkEnv, Observation, Observations
from gym.spaces import Box
from pathlib import Path
import numpy as np


class CartPoleSimulink(SimulinkEnv):
    def __init__(
        self,
        stop_time: float = 15.0,
        step_size: float = 0.001,
        model_debug: bool = False,
    ):
        super().__init__(
            model_path=Path(__file__).parent / "RL_new.slx",
            model_debug=model_debug,
        )

        self.stop_time = stop_time

        self.action_space = Box(low=0, high=120, shape=(4,), dtype=np.float32)

        self.observations = Observations([
            Observation(f"state_{i}", -np.inf, np.inf, "", None) for i in range(9)
        ])

        self.state = self.observations.initial_state

        self.set_model_parameter("StopTime", stop_time)
        self.set_workspace_variable("step_size", step_size)

    def reset(self):
        #初始化状态
        self.observations.initial_state = np.array([0, 0.313443048275326, 4.02751184646594e-17, 0, 66.1102378106091, 66.1102378106091, 66.1102378106091, 66.1102378106091, 0])

        self.state = self.observations.initial_state
        super()._reset()
        return self.state
    


    def compute_r2(self, psi_dot,tar_yaw):
       
        rAVz = -abs(psi_dot-tar_yaw)
        
        return rAVz

     def step(self, action):
        action = np.asarray(action, dtype=np.float32)

        try:
            # 注意：sim_step 的返回顺序可能是 state, sim_time, truncated, terminated, target_yaw
            # 你的 sim_step 中应该是：state, simulation_time, truncated, terminated, target_yaw
            state, sim_time, truncated, terminated, target_yaw = self.sim_step(action)

            # 【核心修改】在这里检查 sim_step 返回的 terminated/truncated 标志
            # 如果 sim_step 内部检测到仿真结束（通过 recv_data == []），它应该会返回 
            # is_truncated = True 并且状态/时间等数据可能无效
            
            # 如果 sim_step 已经通过设置 truncated=True 告知我们仿真结束了
            if truncated:
                # 仿真已结束（被截断）
                #  - 这里的state、sim_time、target_yaw 是 sim_step 返回的，可能是最后一个有效值，
                #    或者由于空数据可能是初始值
                #  - We already know terminated might be false (if it was truncated not terminated normally)
                #  - For RL algorithms, often we use the final state before reset for reward, 
                #    but it's safe to assume if truncated, episode ends.
                
                done = True # 仿真结束了
                
                # 奖励可以根据最后一次有效状态来计算，或者根据sim_step返回的value
                # 这里直接用sim_step返回的target_yaw和state[0] (如果state不是空的话)
                if state is not None and len(state) > 0:
                    reward = self.compute_r2(state[0], target_yaw)
                else:
                    # 如果state是None或空，奖励就为0或者一个设定的惩罚
                    reward = 0.0 # 或者一个小的负值，鼓励提前知道结束
                
                # info 字典可以包含为什么结束了
                info = {"simulation time [s]": sim_time, "reason": "truncated"}
                
                # 当回合结束时，不需要再继续训练，直接返回
                return state, reward, done, info
            
            # --- 如果仿真没有结束 (truncated = False) ---
            self.state = state  # 更新全局状态，这是当前有效状态
            # 假设sim_step也可能通过terminated来提前终止
            done = terminated or (sim_time >= self.stop_time)

            # 计算奖励（仅在仿真进行中时才有意义）
            # 确保 state 有效，通常 state[0] 是一个关键的观察值
            if state is not None and len(state) > 0:
                reward = self.compute_r2(state[0], target_yaw)
            else:
                # 如果sim_step返回了奇怪的值（例如truncated=False但state=None）
                reward = 0.0 
                print("step received invalid state for reward calculation, returning 0 reward.")

            info = {"simulation time [s]": sim_time}
            if terminated:
                info["reason"] = "terminated"
            elif sim_time >= self.stop_time:
                info["reason"] = "max_time_reached"

            return state, reward, done, info

        except Exception as e:
            # 如果 sim_step 抛出了我们没有预料到的其他类型的异常
            print(f"仿真执行出现意外异常：{e}")
            # 我们可以假定此时仿真已无法继续，强制设置done=True
            self.state = self.state # 保留上一个状态，或者置为None/默认值
            # 这里的 reward 可以是根据最后有效状态计算，或设置为0
            if self.state is not None and len(self.state) > 0:
                reward = self.compute_r2(self.state[0], target_yaw) # target_yaw 在异常时可能无效，要小心
            else:
                reward = 0.0 # 假设异常时 reward 是0

            # done=True 告诉 RL 框架需要重置
            done = True
            info = {"simulation time [s]": "N/A", "reason": "exception", "error_message": str(e)}
            return self.state, reward, done, info
