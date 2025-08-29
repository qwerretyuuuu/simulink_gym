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
            state, sim_time, terminated, truncated,target_yaw = self.sim_step(action)
        except Exception as e:
            print("仿真执行失败：", e)
            raise

        self.state = state
        done = terminated or truncated or (sim_time >= self.stop_time)

        reward = self.compute_r2(state[0],target_yaw)

        info = {"simulation time [s]": sim_time}
        return state, reward, done, info
