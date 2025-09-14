#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从QuickStart.ipynb转换而来的Python脚本
包含HVAC系统控制的完整示例
"""


# ====================
# 代码单元格 2
# ====================
import os
# 设置工作目录
cwd = os.getcwd()
print(f"当前工作目录: {cwd}")
# 如果需要切换目录，请取消注释下面的行
# os.chdir("D:\RL\HVAC\BEAR-main")

# ====================
# 代码单元格 3
# ====================
# 添加当前目录到Python路径
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from Env.env_building import BuildingEnvReal
from Controller.MPC_Controller import MPCAgent
from Controller.DataCenterEnergyController import DataCenterEnergyController
from Utils.utils_building import ParameterGenerator, get_user_input
import numpy as np
import functools
import datetime
import os
import time
from collections import deque
import matplotlib.pyplot as plt

# ====================
# 代码单元格 4
# ====================
env = get_user_input()
numofhours=24
#Initialize
env.reset()
for i in range(numofhours):
    a = env.action_space.sample()#Randomly select an action
    obs, r, terminated, truncated, _ = env.step(a)#Return observation and reward
RandomController_state=env.statelist #Collect the state list
RandomController_action=env.actionlist #Collect the action list
env._get_info()

# ====================
# 代码单元格 5
# ====================
RandomController_action[0]

# ====================
# 代码单元格 6
# ====================
def my_custom_reward_function(self, state, action, error, state_new):
    # This is your default reward function
    # Initialize the reward
    reward = 0
    self.co2_rate=0.01
    self.temp_rate=0.01

    # Desired temperature range
    lower_temp = 18
    upper_temp = 22

    # Calculate the contribution of action to the reward
    action_contribution = LA.norm(action, 2) * self.q_rate
    reward -= action_contribution

    # Calculate the contribution of error to the reward
    error_contribution = LA.norm(error, 2) * self.error_rate
    reward -= error_contribution

    # Calculate the contribution of temperature deviation to the reward
    temp_deviation = np.sum(np.maximum(0, state_new - upper_temp) + np.maximum(0, lower_temp - state_new)) * self.temp_rate
    reward -= temp_deviation

    # Calculate the contribution of CO2 emissions to the reward
    co2_emission = LA.norm(action, 2) * self.co2_rate
    reward -= co2_emission

    self._reward_breakdown['action_contribution'] -= action_contribution
    self._reward_breakdown['error_contribution'] -= error_contribution
    self._reward_breakdown['temp_deviation'] -= temp_deviation
    self._reward_breakdown['co2_emission'] -= co2_emission
    return reward

# ====================
# 代码单元格 7
# ====================
Parameter=ParameterGenerator('OfficeSmall','Hot_Dry','Tucson')  #Description of ParameterGenerator in bldg_utils.py
#Create environment
env = BuildingEnvReal(Parameter)
numofhours=24
#Initialize with user-defined indoor temperature
env.reset(options={'T_initial':np.array([18.24489859, 18.58710076, 18.47719682, 19.11476084, 19.59438163,15.39221207])})
for i in range(numofhours):
    a = env.action_space.sample()#Randomly select an action
    obs, r, terminated, truncated, _ = env.step(a)#Return observation and reward
RandomController_state=env.statelist #Collect the state list
RandomController_action=env.actionlist #Collect the action list
env._get_info()

# ====================
# 代码单元格 8
# ====================
obs_dim = env.observation_space.shape[0]
print("Size of State Space ->  {}".format(obs_dim))
action_dim = env.action_space.shape[0]
print("Size of Action Space ->  {}".format(action_dim))
upper_bound = env.action_space.high[0]
lower_bound = env.action_space.low[0]
print("Max Value of Action ->  {}".format(upper_bound))
print("Min Value of Action ->  {}".format(lower_bound))
print('Sample State :', RandomController_state[0])
print('Sample Action :', RandomController_action[0])

# ====================
# 代码单元格 9
# ====================
obs

# ====================
# 代码单元格 10
# ====================

numofhours=24
env.reset()
a=env.action_space.sample()
for i in range(numofhours):

    a = [0,0,0,0,0,0]
    obs, r, terminated, truncated, _ = env.step(a)
plt.plot(np.array(env.statelist)[:,:7])
plt.title('Our Model Temperature')

plt.xlabel('hours')
plt.ylabel('Celsius')

plt.legend(['South','East','North','West','Core','Plenum','Outside'],loc='lower right')
plt.show()
plt.plot(np.sum(np.abs(np.array(env.actionlist)),1))

plt.title('Our Model Power')
plt.xlabel('hours')
plt.ylabel('Watts')
plt.show()

# ====================
# 代码单元格 11
# ====================
print('zone temerature at 1 a.m. :', np.array(env.statelist)[1,:6])

# ====================
# 代码单元格 12
# ====================
agent = MPCAgent(env,
                gamma=env.gamma,
                safety_margin=0.96, planning_steps=10)
env.reset()
numofhours=24
reward_total=0
for i in range(numofhours):
    a,s = agent.predict(env)
    obs, r, terminated, truncated, _ = env.step(a)
    reward_total+=r
print('total reward is: ',reward_total)
plt.plot(np.array(env.statelist)[:,:7])
plt.title('Our Model Temperature')

plt.xlabel('hours')
plt.ylabel('Celsius')
plt.legend(['South','East','North','West','Core','Plenum','Outside'],loc='lower right')
plt.show()
plt.plot(np.sum(np.abs(np.array(env.actionlist)),1))
plt.title('Our Model Power')
plt.xlabel('hours')
plt.ylabel('Watts')
plt.show()
MPCstate=env.statelist
MPCaction=env.actionlist

# ====================
# 代码单元格 13
# ====================
# 使用机房空调节能控制器（基于PDF中的算法）
# 这个控制器实现了区间群控节能策略
agent = DataCenterEnergyController(env,
                                 gamma=env.gamma,
                                 safety_margin=0.96,
                                 planning_steps=10,
                                 alpha=0.1,  # 距离经验参数
                                 beta=0.05,  # 空调运行时间经验参数
                                 temp_tolerance=1.0)  # 温度容忍度

env.reset()
numofhours=24
reward_total=0

print("开始使用机房空调节能控制器进行24小时控制...")
print("算法特点：")
print("1. 基于温度传感器的空调开关控制策略")
print("2. 测温点敏感度分析与影响力计算")
print("3. 区间群控节能优化")

for i in range(numofhours):
    a,s = agent.predict(env)
    obs, r, terminated, truncated, _ = env.step(a)
    reward_total+=r

    # 每6小时显示一次控制统计信息
    if (i+1) % 6 == 0:
        stats = agent.get_control_statistics()
        print(f"第{i+1}小时 - 节能效果: {stats['energy_savings']:.1f}%, 平均能耗: {stats['avg_energy']:.2f}")

print('total reward is: ',reward_total)

# 显示最终控制统计信息
final_stats = agent.get_control_statistics()
print(f"\n=== 机房空调节能控制器性能统计 ===")
print(f"总节能效果: {final_stats['energy_savings']:.1f}%")
print(f"平均能耗: {final_stats['avg_energy']:.2f}")
print(f"平均温度: {final_stats['avg_temp']:.1f}°C")
print(f"平均占用率: {final_stats['avg_occupancy']:.1f}")
print(f"温度控制区间: [{final_stats['temp_range'][0]:.1f}, {final_stats['temp_range'][1]:.1f}]°C")
print(f"影响力矩阵已更新: {final_stats['influence_matrix_updated']}")

# 获取影响力分析
influence_analysis = agent.get_influence_analysis()
print(f"\n=== 影响力分析 ===")
print(f"制冷量配置: {influence_analysis['cooling_capacity']}")
print(f"运行时间历史: {influence_analysis['runtime_history']}")

# 绘制温度曲线
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(np.array(env.statelist)[:,:7])
plt.title('机房空调节能控制器温度控制效果')
plt.xlabel('hours')
plt.ylabel('Celsius')
plt.legend(['South','East','North','West','Core','Plenum','Outside'],loc='lower right')
plt.grid(True)

# 绘制能耗曲线
plt.subplot(2, 1, 2)
plt.plot(np.sum(np.abs(np.array(env.actionlist)),1))
plt.title('机房空调节能控制器能耗控制效果')
plt.xlabel('hours')
plt.ylabel('Watts')
plt.grid(True)
plt.tight_layout()
plt.show()

# 保存结果用于比较
DataCenter_state=env.statelist
DataCenter_action=env.actionlist

# ====================
# 代码单元格 14
# ====================
# 对比分析：传统MPC vs 机房空调节能控制器
print("=== 两种控制器特性对比 ===")
print("\n1. 传统MPC控制器：")
print("   - 基于模型预测控制")
print("   - 优化目标：温度跟踪 + 能耗控制")
print("   - 特点：稳定可靠，但节能效果有限")

print("\n2. 机房空调节能控制器（DataCenterEnergyController）：")
print("   - 基于PDF算法的区间群控节能")
print("   - 测温点敏感度分析与影响力计算")
print("   - 基于温度传感器的空调开关控制策略")
print("   - 特点：专门针对机房环境优化")

# 性能对比（如果有数据的话）
controllers_data = {}

if 'MPCaction' in locals():
    controllers_data['MPC'] = {
        'energy': np.sum(np.abs(MPCaction)),
        'name': '传统MPC'
    }

if 'DataCenter_action' in locals():
    controllers_data['DataCenter'] = {
        'energy': np.sum(np.abs(DataCenter_action)),
        'name': '机房空调节能控制器'
    }

if len(controllers_data) > 1:
    print(f"\n=== 能耗对比分析 ===")
    baseline_energy = None
    for key, data in controllers_data.items():
        if baseline_energy is None:
            baseline_energy = data['energy']
            print(f"{data['name']}: {data['energy']:.2f} (基准)")
        else:
            savings = (baseline_energy - data['energy']) / baseline_energy * 100
            print(f"{data['name']}: {data['energy']:.2f} (节能 {savings:.1f}%)")

print(f"\n=== 推荐使用场景 ===")
print("• 传统MPC：适用于对稳定性要求高的场景")
print("• 机房空调节能控制器：专门适用于机房、数据中心等需要精确温度控制的场景")

print(f"\n=== 算法优势总结 ===")
print("机房空调节能控制器的优势：")
print("1. 基于实际机房环境特点设计")
print("2. 考虑空调与测温点的距离影响")
print("3. 实现区间群控，避免过度调节")
print("4. 结合温度传感器实时反馈")
print("5. 专门针对机房节能优化")

# ====================
# 代码单元格 15
# ====================
print('zone temerature:', np.array(env.statelist)[0,:6])

# ====================
# 代码单元格 16
# ====================
plt.plot(np.array(env.actionlist))
plt.legend(['South','East','North','West','Core','Plenum','Outside'],loc='lower right')
#plt.ylim([0,50])
plt.show()

# ====================
# 代码单元格 17
# ====================
from stable_baselines3 import PPO ,DQN,DDPG
from stable_baselines3.common.logger import configure
from stable_baselines3.ppo import MlpPolicy
# from stable_baselines.bench import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_util import make_vec_env

# ====================
# 代码单元格 18
# ====================

seed=0
env.reset()
set_random_seed(seed=seed)
model = PPO(MlpPolicy, env, verbose=1)
rewardlist=[]
action_record=[]

for i in range(100):
  model.learn(total_timesteps=1000)
  rw=0
  vec_env = model.get_env()
  obs = vec_env.reset()
  for i in range(24):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    rw+=rewards
  print(rw/24)
  rewardlist.append(rw/24)
  action_record.append(np.array(env.actionlist).sum(axis=1))
print("################TRAINING is Done############")
model.save("PPO_quick")

# ====================
# 代码单元格 19
# ====================

model = PPO(MlpPolicy, env, verbose=1)
vec_env = model.get_env()
model = PPO.load("PPO_quick")
obs = vec_env.reset()
print("Initial observation", obs)

for i in range(24):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
plt.plot(np.array(env.statelist)[:,:7])
plt.title('Our Model Temperature')

plt.xlabel('hours')
plt.ylabel('Celsius')
plt.legend(['South','East','North','West','Core','Plenum','Outside'],loc='lower right')
plt.show()
plt.plot(np.sum(np.abs(np.array(env.actionlist)),1))
plt.title('Our Model Power')
plt.xlabel('hours')
plt.ylabel('Watts')
plt.show()

# ====================
# 代码单元格 20
# ====================
plt.title('Quick PPO training')
plt.plot(rewardlist)
plt.xlabel('episode')
plt.ylabel('reward')
plt.show()

# ====================
# 代码单元格 21
# ====================
env.close()