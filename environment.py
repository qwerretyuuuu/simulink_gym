import os
import threading
from pathlib import Path
from typing import Any, Union
import sys
sys.path.append(r'C:\workspace\ai-vdc\XM_MS11_FMU\XM_MS11_FMU_yn\matlab_install\Lib\site-packages')

from pathlib import PurePosixPath
import gymnasium as gym
import matlab.engine
import numpy as np

from . import SIMULINK_BLOCK_LIB_PATH, logger
from .observations import Observations
from .utils import CommSocket
import time


class SimulinkEnv(gym.Env):
    """Wrapper class for using Simulink models through the Gym interface."""

    # Observations to be defined in child class:
    _observations: Observations

    def __init__(
        self,
        model_path: str,
        model_debug: bool = False,
    ):
        
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            # Try as relative path:
            self.model_path = Path(os.path.abspath(model_path))
            if not self.model_path.exists():
                raise ValueError(f"Could not find model under {self.model_path}")
        self.model_dir = self.model_path.parent
        self.env_name = self.model_path.stem
        self.simulation_time = 0
        self.state = None
        self.model_debug = model_debug
        self.time = 0

        # Already prepared replacement for the done flag for Gym/Gymnasium>=0.26.0:
        self.terminated = True
        self.truncated = True


        logger.info("Starting Matlab engine")

        self.matlab_engine = matlab.engine.start_matlab()
        fmi_kit_path = r'C:\CSSim\External_libs\FMIKit\V2.7'
        self.matlab_engine.addpath(fmi_kit_path)

        # 添加 CarSim solver 路径
        self.matlab_engine.eval("addpath(genpath('C:/Program Files (x86)/CarSim2024.3_Prog/Programs/solvers'))", nargout=0)

        # 初始化 FMIKit（如果使用 FMU）
        self.matlab_engine.eval("FMIKit.initialize();", nargout=0)

        # 启动 CarSim COM 服务
        self.matlab_engine.eval("h_carsim = actxserver('CarSim.Application');", nargout=0)

        # 运行 CarSim（RunButtonClick(2) 启动仿真）
        self.matlab_engine.eval("h_carsim.RunButtonClick(2);", nargout=0)

        
        logger.info("成功执行 FMIKit.initialize()")




    def __del__(self):
        """Deletion of environment needs to also quit the Matlab engine."""
        # self.close()
        # Close matlab engine:
        if self.matlab_engine:
            self.matlab_engine.quit()

    @property
    def observations(self):
        """
        Getter method for observations.

        Returns: Observations object defining the observations
        """
        return self._observations

    @observations.setter
    def observations(self, observations: Observations):
        """
        Setter method for observations.

        Also sets the necessary observation space.

        Args:
            observations: Observations object defining the observations
        """
        self._observations = observations
        self.observation_space = self._observations.space

    def _reset(self):

        self.state = self.set_initial_values()

        print("加载simulink环境")
        self.matlab_engine.load_system('RL_new')#加载simulink环境
        self.matlab_engine.set_param('RL_new', 'StopTime', str(15 + 1), nargout=0)

        self.matlab_engine.set_param(f'{self.env_name}/Q1', 'value', str(0), nargout=0)
        self.matlab_engine.set_param(f'{self.env_name}/Q2', 'value', str(0), nargout=0)
        self.matlab_engine.set_param(f'{self.env_name}/Q3', 'value', str(0), nargout=0)
        self.matlab_engine.set_param(f'{self.env_name}/Q4', 'value', str(0), nargout=0)
        self.matlab_engine.set_param(f'{self.env_name}/pause_time', 'value', str(0.01), nargout=0)

        self.matlab_engine.set_param(self.env_name, 'SimulationCommand', 'start', nargout=0)

        self.truncated = False
        self.terminated = False

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:

        raise NotImplementedError

    def sim_step(self, action) -> tuple[np.ndarray, float, bool, bool]:

        # Check validity of action
        if not self.action_space.contains(action):
            raise ValueError(f"Action {action} not in action space.")

        a = time.time()

        # Send action to Simulink
        self.matlab_engine.set_param(f'{self.env_name}/Q1', 'value', str(np.array(action[0])), nargout=0)
        self.matlab_engine.set_param(f'{self.env_name}/Q2', 'value', str(np.array(action[1])), nargout=0)
        self.matlab_engine.set_param(f'{self.env_name}/Q3', 'value', str(np.array(action[2])), nargout=0)
        self.matlab_engine.set_param(f'{self.env_name}/Q4', 'value', str(np.array(action[3])), nargout=0)

        pause_time = 0.01 + self.time
        self.matlab_engine.set_param(f'{self.env_name}/pause_time', 'value', str(pause_time), nargout=0)

        # Start or continue simulation
        status = self.matlab_engine.get_param(self.env_name, 'SimulationStatus')

        if status == 'paused' :
            self.matlab_engine.set_param(self.env_name, 'SimulationCommand', 'continue', nargout=0)

        # Receive data
        recv_data = self.matlab_engine.eval('out.state.Data')
        recv_data = np.array(recv_data, dtype=float)[-1,:]
        
        self.time = self.matlab_engine.eval('out.time.Data')

        if recv_data is None or len(recv_data) == 0:
            self.truncated = True
            print("接收到空消息")

        else:
            self.state = recv_data[:-1]
            self.simulation_time = self.time
            self.target_yaw = recv_data[-1]


        print("耗时：",time.time()-a,"\n")

        return self.state, self.simulation_time, self.truncated, self.terminated,self.target_yaw


    def step(
        self, action
    ) -> tuple[np.ndarray, Union[int, float], bool, bool, dict[str, Any]]:
        """
        Method required by the Gym interface to be implemented by the child class.

        The child method is supposed to call sim_step().

        Args:
            action: action to be executed at the beginning of next simulation step,
                needs to match the defined action space

        Returns:
            state: current state of the environment (according to the observation space)
            reward: numerical reward signal from the environment for reaching current
                state
            terminated: flag indicating termination of the episode
            truncated: flag indicating truncation of the episode
            info:  dict of auxiliary information, e.g. simulation time
        """
        raise NotImplementedError


    def set_initial_values(self):
        """
        Set the initial values of the state/observations.

        Used for resetting the environment.

        Returns:
            initial state according to observation space
        """
        try:
            # Functionality only available if not in debug mode:
            if not self.model_debug:
                self.observations.reset_values()
        except AttributeError:
            raise AttributeError("Environment observations not defined")

        return self.observations.initial_state

    def _send_stop_signal(self):
        """Method for sending the stop signal to the simulation."""
        set_values = np.zeros(self.action_space.shape)
        self.send_data(set_values, stop=True)

    def stop_simulation(self):
        """Method for stopping the simulation."""
        if self.model_debug or self.simulation_thread.is_alive():
            try:
                self._send_stop_signal()
            except Exception:
                # Connection already lost
                logger.info(
                    "Stop signal could not be sent, connection probably already dead"
                )
            else:
                # Clear receive data queue:
                _ = self.recv_socket.receive()
            finally:
                if not self.model_debug:
                    self.simulation_thread.join()

        self.truncated = True


    def render(self):
        """
        Render method recommended by the Gym interface.

        Since Simulink models don't share a common representation suitable for rendering
        such a method is not possible to implement.
        """
        logger.info("Rendering not supported for Simulink models.")


#=====================================================================================================

# import os
# import sys
# sys.path.append(r'C:\workspace\ai-vdc\XM_MS11_FMU\XM_MS11_FMU_yn\matlab_install\Lib\site-packages')

# from pathlib import PurePosixPath

# import matlab.engine
# import gym
# from . import logger, SIMULINK_BLOCK_LIB_PATH
# import threading
# import numpy as np
# from typing import Union
# from pathlib import Path
# from .observations import Observations
# from .utils import CommSocket
# import time


# class SimulinkEnv(gym.Env):
#     """通过 Gym 接口封装 Simulink 模型的环境基类"""

#     # 由子类定义具体的观测变量
#     _observations: Observations

#     def __init__(
#         self,
#         model_path: str,
#         send_port: int = 42313,
#         recv_port: int = 42312,
#         model_debug: bool = False,
#     ):
#         """
#         Simulink 环境基类，实现 Gym 的接口

#         参数：
#             model_path: str
#                 Simulink 模型文件路径
#             send_port: int, 默认 42313
#                 发送数据的 TCP/IP 端口
#             recv_port: int, 默认 42312
#                 接收数据的 TCP/IP 端口
#             model_debug: bool, 默认 False
#                 是否开启模型调试模式（不启动 Matlab 引擎）
#         """
#         self.model_path = Path(model_path)
#         # 判断路径是否存在，不存在则尝试绝对路径
#         if not self.model_path.exists():
#             self.model_path = Path(os.path.abspath(model_path))
#             if not self.model_path.exists():
#                 raise ValueError(f"无法找到模型文件，路径：{self.model_path}")
#         self.model_dir = self.model_path.parent
#         self.env_name = self.model_path.stem  # 模型文件名（无扩展名）
#         self.simulation_time = 0
#         self.state = None
#         self.model_debug = model_debug

#         # 用于替代 Gym/Gymnasium 0.26.0 及以上版本的 done 标志
#         self.terminated = True
#         self.truncated = True

#         # 创建通信用的 TCP/IP 套接字（接收和发送）
#         self.recv_socket = CommSocket(recv_port, "recv_socket")
#         self.send_socket = CommSocket(send_port, "send_socket")

#         if not self.model_debug:
#             # 非调试模式下，启动模拟线程和 Matlab 引擎
#             self.simulation_thread = threading.Thread()
#             logger.info("启动 Matlab 引擎")
#             matlab_started = False
#             start_trials = 0
#             # 尝试启动 Matlab 引擎，最多重试3次
#             while not matlab_started and start_trials < 3:
#                 try:
#                     self.matlab_engine = matlab.engine.start_matlab()
#                 except matlab.engine.RejectedExecutionError:
#                     start_trials += 1
#                     logger.error("启动 Matlab 引擎失败，重试中...")
#                 else:
#                     matlab_started = True
#                     logger.info("将组件路径添加至 Matlab 路径")
#                     # 添加 Simulink 模块库路径
#                     self.matlab_path = self.matlab_engine.addpath(
#                         str(SIMULINK_BLOCK_LIB_PATH)
#                     )
#                     # 添加模型所在目录路径
#                     self.matlab_path = self.matlab_engine.addpath(
#                         str(self.model_dir.absolute())
#                     )


#                     fmi_kit_path = r'C:\CSSim\External_libs\FMIKit\V2.7'
#                     self.matlab_engine.addpath(fmi_kit_path)

#                     # 初始化 FMIKit
#                     try:
#                         # 添加 CarSim solver 路径
#                         self.matlab_engine.eval("addpath(genpath('C:/Program Files (x86)/CarSim2024.3_Prog/Programs/solvers'))", nargout=0)

#                         # 初始化 FMIKit（如果使用 FMU）
#                         self.matlab_engine.eval("FMIKit.initialize();", nargout=0)

#                         # 启动 CarSim COM 服务
#                         self.matlab_engine.eval("h_carsim = actxserver('CarSim.Application');", nargout=0)

#                         # 运行 CarSim（RunButtonClick(2) 启动仿真）
#                         self.matlab_engine.eval("h_carsim.RunButtonClick(2);", nargout=0)

                        
#                         logger.info("成功执行 FMIKit.initialize()")
#                     except matlab.engine.MatlabExecutionError as e:
#                         logger.error("执行 FMIKit.initialize() 失败：")
#                         logger.error(str(e))

#                     # 创建仿真输入对象
#                     logger.info(
#                         f"为模型 {self.env_name}.slx 创建仿真输入对象"
#                     )
#                     self.sim_input = self.matlab_engine.Simulink.SimulationInput(
#                         self.env_name
#                     )
#             if not matlab_started and start_trials >= 3:
#                 raise RuntimeError("启动 Matlab 引擎失败，尝试次数超过限制。")
#         else:
#             # 调试模式下不需要以下变量
#             self.simulation_thread = None
#             self.matlab_engine = None
#             self.matlab_path = None
#             self.sim_input = None

#     def __del__(self):
#         """删除环境时，需要关闭 Matlab 引擎和仿真"""
#         self.close()
#         if self.matlab_engine:
#             self.matlab_engine.quit()

#     @property
#     def observations(self):
#         """观测变量的 getter"""
#         return self._observations

#     @observations.setter
#     def observations(self, observations: Observations):
#         """观测变量的 setter，同时设置 observation_space"""
#         self._observations = observations
#         self.observation_space = self._observations.space

#     def _reset(self):
#         """
#         环境通用的 reset 行为

#         该方法会停止正在运行的仿真，关闭并重开通信套接字，重新启动仿真
#         """
#         if self.simulation_thread and self.simulation_thread.is_alive():
#             self.stop_simulation()

#         self.close_sockets()
#         self.open_sockets()

#         # 设置初始状态值
#         self.state = self.set_initial_values()

#         if not self.model_debug:
#             # 启动仿真线程，运行 Simulink 模型仿真
#             aa = time.time()
#             self.simulation_thread = threading.Thread(
#                 name="sim thread", target=self.matlab_engine.sim, args=(self.sim_input,)
#             )
#             self.simulation_thread.start()

#             print("Simulink运行时间：",time.time()-aa)

#         # 等待 TCP 连接建立
#         self.send_socket.wait_for_connection()
#         self.recv_socket.wait_for_connection()

#         # 重置终止和截断标志
#         self.truncated = False
#         self.terminated = False

#     def reset(self):
#         """
#         必须由子类实现的方法，用于重置环境

#         子类应调用 _reset() 并返回初始状态
#         """
#         raise NotImplementedError
    
#     def receive_current_data(self):
#         recv_data = self.recv_socket.receive()
#         return recv_data

#     def sim_step(self, action):
#         """
#         与 Simulink 模型通信，执行一步仿真

#         参数:
#             action: 要执行的动作，必须符合动作空间

#         返回:
#             state: 当前环境状态（观测空间格式）
#             simulation_time: 当前仿真时间（秒）
#             truncated: 是否被截断（非终止的强制结束）
#             terminated: 是否达到终止状态
#         """
#         if self.model_debug or self.simulation_thread.is_alive():
#             # 验证动作是否有效
#             if not self.action_space.contains(action):
#                 raise ValueError(f"动作 {action} 不在动作空间内。")
            
# #------------------------------------------------------------------------------------
#             # 发送动作给 Simulink
#             action = np.array(action, dtype=np.float32)  # 转换为 NumPy 数组并指定类型
#             self.send_data(np.array(action))
# #------------------------------------------------------------------------------------
            
# #------------------------------------------------------------------------------------
#             # 接收 Simulink 返回的数据
#             recv_data = self.recv_socket.receive()
# #------------------------------------------------------------------------------------


#             if len(recv_data) != 11:
#                 raise ValueError(f"接收到的数据维度无效！实际维度 {len(recv_data)}，应为 11(state + target_yaw + time)")

#             # 如果截断，接收到空消息
#             if not recv_data:
#                 self.truncated = True
#                 print("接收到空消息")
#             else:
#                 # 数据有效，提取状态和仿真时间
#                 self.state = np.array(recv_data[0:-2], ndmin=1, dtype=np.float32)
#                 self.simulation_time = recv_data[-1]
#                 self.target_yaw = recv_data[-2]

#         else:
#             # 仿真线程未启动，不能执行仿真步
#             logger.warn("当前无运行中的仿真，无法执行仿真步。")
#             self.truncated = True

#         return self.state, self.simulation_time, self.truncated, self.terminated,self.target_yaw

#     def step(self, action):
#         """
#         Gym 接口必须实现的方法，子类需要实现具体细节

#         通常应调用 sim_step() 执行仿真一步

#         参数：
#             action：执行的动作，必须符合动作空间

#         返回：
#             state：当前状态（观测空间）
#             reward：当前状态奖励值
#             done：是否结束
#             info：附加信息字典（如仿真时间）
#         """
#         raise NotImplementedError

#     def send_data(self, set_values: np.ndarray, stop: bool = False):
#         """
#         发送数据到 Simulink 模型

#         参数：
#             set_values：动作数据，形状需匹配动作空间
#             stop：是否发送停止信号（默认 False）
#         """
#         if set_values.shape == self.action_space.shape:
#             if self.model_debug or self.simulation_thread.is_alive():
#                 self.send_socket.send_data(set_values, stop)
                
#                 # print(f"[SEND] Action to Simulink: {set_values}, Stop: {stop}")

#             else:
#                 logger.info("当前无仿真运行，无法发送数据。")
#         else:
#             raise Exception(
#                 f"数据形状错误，传入形状 {set_values.shape}，应为 {self.action_space.shape}。"
#             )

#     def set_workspace_variable(self, var: str, value: Union[int, float]):
#         """
#         设置 Simulink 模型工作区变量

#         模型块使用工作区变量时，可通过此方法设置变量值

#         注意：频繁调用会占用大量内存，建议少用

#         参数：
#             var：变量名
#             value：变量值
#         """
#         if not self.model_debug:
#             self.sim_input = self.matlab_engine.setVariable(
#                 self.sim_input, var, float(value), "Workspace", self.env_name
#             )

#     def set_block_parameter(self, path: str, value: Union[int, float]):
#         """
#         设置 Simulink 模型块参数

#         参数：
#             path：块路径（使用 '/' 分隔）
#             value：参数值
#         """
#         if not self.model_debug:
#             # 保持路径分隔符为 '/'，符合 Simulink 块路径格式
#             posix_path = PurePosixPath(path)
#             block_path = str(posix_path.parent)
#             param = str(posix_path.name)  # 取块名
#             value = str(value)
#             self.sim_input = self.matlab_engine.setBlockParameter(
#                 self.sim_input, block_path, param, value
#             )

#     def set_model_parameter(self, param: str, value: Union[int, float]):
#         """
#         设置 Simulink 模型参数

#         注意：频繁调用会占用大量内存，建议少用

#         参数：
#             param：参数名
#             value：参数值
#         """
#         if not self.model_debug:
#             self.sim_input = self.matlab_engine.setModelParameter(
#                 self.sim_input, param, str(value)
#             )

#     def set_initial_values(self):
#         """
#         设置环境初始状态值（用于重置）

#         返回：
#             初始状态，符合观测空间格式
#         """
#         try:
#             if not self.model_debug:
#                 self.observations.reset_values()
#         except AttributeError:
#             raise AttributeError("环境未定义观测变量")

#         return self.observations.initial_state

#     def _send_stop_signal(self):
#         """发送停止信号给仿真，通常发送全零动作加停止标志"""
#         set_values = np.zeros(self.action_space.shape)
#         self.send_data(set_values, stop=True)

#     def stop_simulation(self):
#         """停止仿真，发送停止信号并关闭仿真线程"""
#         if self.model_debug or self.simulation_thread.is_alive():
#             try:
#                 self._send_stop_signal()
#             except Exception:
#                 # 连接已断开
#                 logger.info("停止信号发送失败，连接可能已断开。")
#             else:
#                 # 清空接收队列数据
#                 _ = self.recv_socket.receive()
#             finally:
#                 if not self.model_debug:
#                     self.simulation_thread.join()

#         self.truncated = True

#     def open_sockets(self):
#         """打开发送和接收套接字"""
#         self.recv_socket.open_socket()
#         self.send_socket.open_socket()

#     def close_sockets(self):
#         """关闭发送和接收套接字"""
#         self.recv_socket.close()
#         self.send_socket.close()

#     def close(self):
#         """关闭仿真环境，停止仿真并关闭套接字"""
#         self.stop_simulation()
#         self.close_sockets()

#     def render(self):
#         """
#         Gym 推荐的渲染接口

#         由于 Simulink 模型没有统一的可视化接口，此方法无法实现
#         """
#         pass
