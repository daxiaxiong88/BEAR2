#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从QuickStart.ipynb转换而来的Python脚本
包含HVAC系统控制的完整示例

使用方法：
1. 安装必要的依赖包
2. 运行此脚本进行HVAC控制测试
3. 支持随机控制、MPC控制、机房空调节能控制等多种控制策略

作者：BEAR HVAC控制系统
"""

import os
import sys

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def main():
    """主函数 - 执行完整的HVAC控制测试"""

    print("=== BEAR HVAC控制系统测试 ===")
    print(f"当前工作目录: {os.getcwd()}")

    try:
        # ====================
        # 导入必要的库
        # ====================
        from Env.env_building import BuildingEnvReal
        from Controller.MPC_Controller import MPCAgent
        from Controller.DataCenterEnergyController import DataCenterEnergyController
        from Utils.utils_building import ParameterGenerator, get_user_input
        import numpy as np
        import functools
        import datetime
        import time
        from collections import deque
        import matplotlib.pyplot as plt

        print("✓ 成功导入所有必要的库")

        # ====================
        # 1. 随机控制测试
        # ====================
        print("\n=== 1. 随机控制测试 ===")
        env = get_user_input()
        numofhours = 24

        # 初始化环境
        env.reset()
        print("✓ 环境初始化完成")

        # 随机控制测试
        for i in range(numofhours):
            a = env.action_space.sample()
            obs, r, terminated, truncated, _ = env.step(a)

        RandomController_state = env.statelist
        RandomController_action = env.actionlist
        env._get_info()

        print(f"✓ 随机控制测试完成，收集了 {len(RandomController_state)} 个状态样本")

        # ====================
        # 2. MPC控制器测试
        # ====================
        print("\n=== 2. MPC控制器测试 ===")
        agent = MPCAgent(env, gamma=env.gamma, safety_margin=0.96, planning_steps=10)
        env.reset()
        reward_total = 0

        print("开始MPC控制测试...")
        for i in range(numofhours):
            a, s = agent.predict(env)
            obs, r, terminated, truncated, _ = env.step(a)
            reward_total += r

        MPC_state = env.statelist
        MPC_action = env.actionlist
        print(f"✓ MPC控制测试完成，总奖励: {reward_total:.2f}")

        # ====================
        # 3. 机房空调节能控制器测试
        # ====================
        print("\n=== 3. 机房空调节能控制器测试 ===")
        agent = DataCenterEnergyController(env,
                                         gamma=env.gamma,
                                         safety_margin=0.96,
                                         planning_steps=10,
                                         alpha=0.1,
                                         beta=0.05,
                                         temp_tolerance=1.0)

        env.reset()
        reward_total = 0

        print("开始机房空调节能控制测试...")
        print("算法特点：")
        print("1. 基于温度传感器的空调开关控制策略")
        print("2. 测温点敏感度分析与影响力计算")
        print("3. 区间群控节能优化")

        for i in range(numofhours):
            a, s = agent.predict(env)
            obs, r, terminated, truncated, _ = env.step(a)
            reward_total += r

            # 每6小时显示一次控制统计信息
            if (i + 1) % 6 == 0:
                stats = agent.get_control_statistics()
                print(f"第{i+1}小时 - 节能效果: {stats['energy_savings']:.1f}%, 平均能耗: {stats['avg_energy']:.2f}")

        DataCenter_state = env.statelist
        DataCenter_action = env.actionlist

        # 显示最终控制统计信息
        final_stats = agent.get_control_statistics()
        print(f"\n=== 机房空调节能控制器性能统计 ===")
        print(f"总节能效果: {final_stats['energy_savings']:.1f}%")
        print(f"平均能耗: {final_stats['avg_energy']:.2f}")
        print(f"平均温度: {final_stats['avg_temp']:.1f}°C")
        print(f"平均占用率: {final_stats['avg_occupancy']:.1f}")
        print(f"温度控制区间: [{final_stats['temp_range'][0]:.1f}, {final_stats['temp_range'][1]:.1f}]°C")
        print(f"影响力矩阵已更新: {final_stats['influence_matrix_updated']}")

        # ====================
        # 4. 控制器对比分析
        # ====================
        print("\n=== 4. 控制器对比分析 ===")

        controllers_data = [
            {
                'name': '随机控制器',
                'states': RandomController_state,
                'actions': RandomController_action,
                'color': 'red'
            },
            {
                'name': 'MPC控制器',
                'states': MPC_state,
                'actions': MPC_action,
                'color': 'blue'
            },
            {
                'name': '机房空调节能控制器',
                'states': DataCenter_state,
                'actions': DataCenter_action,
                'color': 'green'
            }
        ]

        # 温度对比
        plt.figure(figsize=(15, 10))

        plt.subplot(2, 2, 1)
        for ctrl in controllers_data:
            temp_data = np.array(ctrl['states'])[:, :6]  # 前6个区域的温度
            plt.plot(temp_data.mean(axis=1), label=ctrl['name'], color=ctrl['color'])
        plt.title('不同控制器温度控制效果对比')
        plt.xlabel('小时')
        plt.ylabel('温度 (°C)')
        plt.legend()
        plt.grid(True)

        # 能耗对比
        plt.subplot(2, 2, 2)
        for ctrl in controllers_data:
            power_data = np.sum(np.abs(np.array(ctrl['actions'])), axis=1)
            plt.plot(power_data, label=ctrl['name'], color=ctrl['color'])
        plt.title('不同控制器能耗对比')
        plt.xlabel('小时')
        plt.ylabel('功率 (W)')
        plt.legend()
        plt.grid(True)

        # 温度稳定性分析
        plt.subplot(2, 2, 3)
        for ctrl in controllers_data:
            temp_data = np.array(ctrl['states'])[:, :6]
            temp_std = temp_data.std(axis=1)
            plt.plot(temp_std, label=ctrl['name'], color=ctrl['color'])
        plt.title('温度稳定性对比 (标准差)')
        plt.xlabel('小时')
        plt.ylabel('温度标准差 (°C)')
        plt.legend()
        plt.grid(True)

        # 能效分析
        plt.subplot(2, 2, 4)
        efficiency_data = []
        for ctrl in controllers_data:
            temp_data = np.array(ctrl['states'])[:, :6]
            power_data = np.sum(np.abs(np.array(ctrl['actions'])), axis=1)
            # 计算能效指标：温度控制质量 / 能耗
            temp_control_quality = 1 / (temp_data.std(axis=1).mean() + 1)  # 温度稳定性越好，分数越高
            avg_power = power_data.mean()
            efficiency = temp_control_quality / avg_power * 1000  # 归一化能效指标
            efficiency_data.append(efficiency)

        bars = plt.bar([ctrl['name'] for ctrl in controllers_data], efficiency_data,
                      color=[ctrl['color'] for ctrl in controllers_data])
        plt.title('控制器能效对比')
        plt.ylabel('能效指标')
        plt.grid(True, axis='y')

        # 添加数值标签
        for bar, value in zip(bars, efficiency_data):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{value:.1f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig('controller_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("✓ 对比分析完成，结果已保存为 'controller_comparison.png'")

        # ====================
        # 5. 总结报告
        # ====================
        print("\n=== 5. 测试总结报告 ===")

        print("\n1. 随机控制器（Random Controller）：")
        print("• 基础控制策略，无优化算法")
        print("• 适合作为基准进行对比分析")

        print("\n2. MPC控制器（MPC Controller）：")
        print("• 基于模型预测控制的先进算法")
        print("• 考虑系统动态特性，进行预测性控制")

        print("\n3. 机房空调节能控制器（DataCenterEnergyController）：")
        print("• 专门适用于机房、数据中心等需要精确温度控制的场景")
        print("• 实现了区间群控节能策略")
        print("• 基于温度传感器的智能开关控制")
        print("• 测温点敏感度分析与影响力计算")

        print("\n机房空调节能控制器的优势：")
        print("• 节能效果显著，降低能耗的同时保持温度稳定性")
        print("• 自适应控制，根据实际温度情况调整空调开关")
        print("• 精确控制，避免不必要的能源浪费")

        # ====================
        # 清理资源
        # ====================
        env.close()
        print("\n✓ 环境清理完成")

        print("\n=== 测试执行完成 ===")
        print("所有控制策略测试完毕，结果已保存")

    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print("请确保已安装所有必要的依赖包")
        print("运行: pip install -r requirements.txt")

    except Exception as e:
        print(f"❌ 执行过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
