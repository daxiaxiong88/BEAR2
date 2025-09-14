import numpy as np

# 尝试导入cvxpy，如果失败则使用替代方案
try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    print("警告: cvxpy未安装，将使用简化的控制策略")
    CVXPY_AVAILABLE = False


class DataCenterEnergyController(object):
    """
    基于机房空调节能算法的高效控制器
    实现区间群控节能策略
    """

    def __init__(self, environment, gamma, safety_margin=0.9,
                 planning_steps=10, alpha=0.1, beta=0.05,
                 temp_tolerance=1.0):
        """
        初始化机房节能控制器

        Args:
            environment: 环境对象
            gamma: 控制参数
            safety_margin: 安全裕度
            planning_steps: 预测步数
            alpha: 距离经验参数
            beta: 空调运行时间经验参数
            temp_tolerance: 温度容忍度
        """
        self.gamma = gamma
        self.safety_margin = safety_margin
        self.planning_steps = planning_steps
        self.alpha = alpha
        self.beta = beta
        self.temp_tolerance = temp_tolerance

        # 环境参数
        self.action_space = environment.action_space
        self.Qlow = environment.Qlow
        self.num_of_state = environment.roomnum
        self.num_of_action = environment.action_space.shape[0]
        self.A_d = environment.A_d
        self.B_d = environment.B_d
        self.temp = environment.OutTemp[environment.epochs]
        self.acmap = environment.acmap
        self.GroundTemp = environment.GroundTemp[environment.epochs]
        self.Occupancy = environment.Occupancy[environment.epochs]
        self.ghi = environment.ghi[environment.epochs]
        self.target = environment.target
        self.spacetype = environment.spacetype

        # 机房节能控制参数
        self.cooling_capacity = np.ones(self.num_of_action)  # 制冷量
        self.runtime_history = np.zeros(self.num_of_action)  # 运行时间
        self.influence_matrix = np.zeros((self.num_of_state,
                                          self.num_of_action))
        self.normalized_influence = np.zeros((self.num_of_state,
                                             self.num_of_action))

        # 温度区间设置
        # 确保target是标量值
        if isinstance(self.target, (list, np.ndarray)):
            self.target = (float(self.target[0]) if len(self.target) > 0
                           else 22.0)
        else:
            self.target = float(self.target)
        
        self.temp_min = self.target - self.temp_tolerance
        self.temp_max = self.target + self.temp_tolerance

        # 控制历史
        self.control_history = []
        self.energy_history = []
        self.temp_history = []

        self.problem = None

    def _distance_decay_function(self, distance, runtime):
        """
        距离和时间衰减函数
        f(d, t) = alpha * d^2 + beta * t
        """
        return self.alpha * distance**2 + self.beta * runtime

    def _calculate_influence_matrix(self):
        """
        计算所有测温点的影响力矩阵
        基于算法1：测温点敏感度分析与影响力计算
        """
        # 模拟距离矩阵（实际应用中应该从环境获取真实距离）
        # 这里使用随机距离作为示例
        np.random.seed(42)  # 确保结果可重复
        distance_matrix = np.random.uniform(
            1, 10, (self.num_of_state, self.num_of_action))

        # 计算影响力矩阵
        for i in range(self.num_of_state):
            for j in range(self.num_of_action):
                # 计算空调j对测温点i的影响力
                decay = self._distance_decay_function(
                    distance_matrix[i, j],
                    self.runtime_history[j])
                self.influence_matrix[i, j] = self.cooling_capacity[j] * decay

        # 归一化处理
        for i in range(self.num_of_state):
            total_influence = float(np.sum(self.influence_matrix[i, :]))
            if total_influence > 0:
                self.normalized_influence[i, :] = (
                    self.influence_matrix[i, :] / total_influence)

    def _temperature_based_control_strategy(self, current_temps):
        """
        基于温度传感器的空调开关控制策略
        基于算法2：基于温度传感器的空调开关控制策略
        """
        # 初始化空调开关状态（默认关闭）
        control_action = np.zeros(self.num_of_action)
        
        # 确保current_temps是numpy数组
        if not isinstance(current_temps, np.ndarray):
            current_temps = np.array(current_temps)
        
        # 检查是否有任何温度超过目标区间最大值
        temp_comparison = current_temps >= self.temp_max
        if isinstance(temp_comparison, bool):
            if temp_comparison:
                control_action = np.ones(self.num_of_action)
                return control_action
        else:
            if bool(np.any(temp_comparison)):
                control_action = np.ones(self.num_of_action)
                return control_action
        
        # 如果所有温度都在目标区间内，按影响力升序关闭空调
        for i in range(self.num_of_state):
            temp_i = float(current_temps[i])  # 确保是标量值
            
            # 如果温度低于目标区间最大值，按影响力升序关闭空调
            if temp_i < self.temp_max:
                # 获取测温点i的归一化影响力
                influence_i = self.normalized_influence[i, :]
                
                # 按影响力升序排列（影响力小的先关闭）
                sorted_indices = np.argsort(influence_i)
                
                # 逐步关闭空调
                for j in sorted_indices:
                    if control_action[j] == 1:  # 如果空调是开启状态
                        control_action[j] = 0  # 关闭空调
                        
                        # 模拟温度变化检测
                        # 在实际应用中，这里应该实时检测温度变化
                        # 如果温度上升到目标区间最大值，停止关闭
                        if temp_i >= self.temp_max:
                            control_action = np.ones(self.num_of_action)
                            break
        
        return control_action

    def _optimize_control_with_mpc(self, environment):
        """
        结合MPC优化控制策略
        在温度控制的基础上，使用MPC进行精细调节
        """
        # 获取基于温度的控制动作
        temp_based_action = self._temperature_based_control_strategy(
            environment.state[:self.num_of_state])
        
        # 如果cvxpy不可用，使用简化的控制策略
        if not CVXPY_AVAILABLE:
            print("使用简化的控制策略（cvxpy不可用）")
            # 基于温度偏差的简单控制
            current_temps = environment.state[:self.num_of_state]
            temp_error = current_temps - self.target
            
            # 简单的比例控制
            simple_action = -0.1 * temp_error
            simple_action = np.clip(simple_action, -1.0, 1.0)
            
            # 结合温度控制策略
            # 确保simple_action和temp_based_action形状匹配
            if len(simple_action) != len(temp_based_action):
                # 如果形状不匹配，使用temp_based_action作为主要控制
                combined_action = temp_based_action
            else:
                combined_action = simple_action * temp_based_action
            return combined_action, environment.state[:self.num_of_state]
        
        # 使用MPC进行精细调节
        x0 = cp.Parameter(self.num_of_state, name='x0')
        u_max = cp.Parameter(self.num_of_action, name='u_max')
        u_min = cp.Parameter(self.num_of_action, name='u_min')
        
        x = cp.Variable((self.num_of_state, self.planning_steps + 1), name='x')
        u = cp.Variable((self.num_of_action, self.planning_steps), name='u')
        
        # 设置参数值
        x0.value = environment.state[:self.num_of_state]
        u_max.value = 1.0 * np.ones((self.num_of_action,))
        u_min.value = -1.0 * np.ones((self.num_of_action,))
        
        x_desired = self.target
        
        # 构建约束和目标函数
        obj = 0
        constr = [x[:, 0] == x0]
        
        # 计算当前状态
        avg_temp = float(np.sum(x0.value)) / self.num_of_state
        Meta = self.Occupancy
        self.Occupower = self._calculate_occupancy_power(avg_temp, Meta)
        
        # 构建预测模型
        for t in range(self.planning_steps):
            # 系统动态约束
            constr += [
                x[:, t + 1] == (self.A_d @ x[:, t].T +
                                self.B_d[:, 3:-1] @ u[:, t] +
                                self.B_d[:, 2] * self.temp +
                                self.B_d[:, 1] * self.GroundTemp +
                                self.B_d[:, 0] * self.Occupower +
                                self.B_d[:, -1] * self.ghi),
                u[:, t] <= u_max,
                u[:, t] >= u_min,
            ]
            
            # 温度约束
            temp_lower = x_desired - 2.0
            temp_upper = x_desired + 2.0
            constr += [
                x[:, t] >= temp_lower,
                x[:, t] <= temp_upper
            ]
            
            # 目标函数：温度跟踪 + 能耗控制
            obj += (self.gamma[1] * cp.norm(
                cp.multiply(x[:, t], self.acmap) -
                x_desired * self.acmap, 2) +
                self.gamma[0] * 24 * cp.norm(u[:, t], 2))
        
        # 求解优化问题
        prob = cp.Problem(cp.Minimize(obj), constr)
        
        try:
            prob.solve(solver='ECOS_BB', verbose=False)
            
            if prob.status in ["infeasible", "unbounded"]:
                print(f"MPC优化失败: {prob.status}")
                return temp_based_action, x0.value
            else:
                mpc_action = u.value[:, 0]
                state = x[:, 1].value
                
                # 结合温度控制策略和MPC结果
                # 如果温度控制策略要求关闭某些空调，则相应调整MPC动作
                combined_action = mpc_action * temp_based_action
                
                return combined_action, state
                
        except Exception as e:
            print(f"MPC求解错误: {e}")
            print("使用基于温度的控制策略作为备选方案")
            return temp_based_action, x0.value

    def _calculate_occupancy_power(self, avg_temp, occupancy):
        """计算人员占用功率"""
        return (6.461927 + 0.946892 * occupancy +
                0.0000255737 * occupancy**2 -
                0.0627909 * avg_temp * occupancy +
                0.0000589172 * avg_temp * occupancy**2 -
                0.19855 * avg_temp**2 +
                0.000940018 * avg_temp**2 * occupancy -
                0.00000149532 * avg_temp**2 * occupancy**2)

    def predict(self, environment):
        """
        预测控制主函数
        实现机房空调节能算法
        """
        # 更新环境参数
        self.A_d = environment.A_d
        self.B_d = environment.B_d
        self.temp = float(environment.OutTemp[environment.epochs])
        self.GroundTemp = float(environment.GroundTemp[environment.epochs])
        self.Occupancy = float(environment.Occupancy[environment.epochs])
        self.ghi = float(environment.ghi[environment.epochs])

        # 更新运行时间历史
        self.runtime_history += 1

        # 计算影响力矩阵
        self._calculate_influence_matrix()

        # 获取控制动作
        action, state = self._optimize_control_with_mpc(environment)

        # 更新控制历史
        self.control_history.append(action.copy())
        self.energy_history.append(np.linalg.norm(action))
        self.temp_history.append(
            np.mean(environment.state[:self.num_of_state]))

        # 保持历史记录长度
        if len(self.control_history) > 100:
            self.control_history = self.control_history[-100:]
            self.energy_history = self.energy_history[-100:]
            self.temp_history = self.temp_history[-100:]

        # 根据空间类型调整动作
        if self.spacetype == 'continuous':
            return action, state
        else:
            return (action * 100 - self.Qlow * 100).astype(int), state

    def get_energy_savings(self):
        """计算节能效果"""
        if len(self.energy_history) < 10:
            return 0.0

        current_avg = np.mean(self.energy_history[-10:])
        baseline_energy = 1.0
        savings = (baseline_energy - current_avg) / baseline_energy * 100
        return max(0, savings)

    def get_control_statistics(self):
        """获取控制统计信息"""
        stats = {
            'energy_savings': self.get_energy_savings(),
            'avg_energy': (np.mean(self.energy_history)
                           if self.energy_history else 0),
            'avg_temp': (np.mean(self.temp_history)
                         if self.temp_history else 0),
            'avg_occupancy': (np.mean(self.Occupancy)
                              if hasattr(self, 'Occupancy') else 0),
            'control_steps': len(self.energy_history),
            'influence_matrix_updated': True,
            'temp_range': [self.temp_min, self.temp_max]
        }
        return stats

    def get_influence_analysis(self):
        """获取影响力分析结果"""
        return {
            'influence_matrix': self.influence_matrix.copy(),
            'normalized_influence': self.normalized_influence.copy(),
            'cooling_capacity': self.cooling_capacity.copy(),
            'runtime_history': self.runtime_history.copy()
        }
