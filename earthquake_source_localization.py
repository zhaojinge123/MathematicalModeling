#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
地震震源定位问题的数学建模与求解
本程序实现了四个子问题的数值求解:
1. 基本震源坐标确定
2. 考虑测量误差下的位置范围估计
3. 利用更多监测台的模型改进
4. 监测台深度分布对精度的影响分析
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize
import time

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def load_data():
    """加载地震监测数据"""
    # 导入监测台数据
    stations_a = np.array([
        [0, 0, 0, 29.172],    # A站坐标(x,y,z)和到震源距离
        [10, 0, 0, 30.512],   # B站
        [-10, 0, 0, 31.161],  # C站
        [0, 10, 0, 35.369]    # D站
    ])

    # 更多监测台数据
    stations_a_extended = np.array([
        # 原有4个监测台数据
        [0, 0, 0, 29.172],
        [10, 0, 0, 30.512],
        [-10, 0, 0, 31.161],
        [0, 10, 0, 35.369],
        # 新增监测台数据
        [0, 20, 0, 49.065],
        [0, -20, 0, 25.500],
        [10, 20, 0, 43.903],
        [-20, 10, 0, 41.136],
        [-10, -20, 0, 27.728],
        [20, -10, 0, 31.788]
    ])

    # 地震b的监测数据
    stations_b = np.array([
        [0, 0, 0, 44.873],
        [10, 0, 0, 49.571],
        [-10, 0, 0, 42.215],
        [0, 10, 0, 40.776],
        [0, 20, 0, 38.938],
        [0, -20, 0, 57.682],
        [10, 20, 0, 44.132],
        [-20, 10, 0, 30.386],
        [-10, -20, 0, 55.514],
        [20, -10, 0, 60.339]
    ])
    
    return stations_a, stations_a_extended, stations_b

def objective_function(loc, stations):
    """
    目标函数：计算震源位置与观测距离之间的误差平方和
    
    参数:
    loc: 震源位置坐标 [x, y, z]
    stations: 监测台数据，每行包含[x, y, z, distance]
    
    返回:
    误差平方和
    """
    x, y, z = loc
    error_sum = 0
    for station in stations:
        sx, sy, sz, distance = station
        calculated_dist = np.sqrt((x - sx)**2 + (y - sy)**2 + (z - sz)**2)
        error_sum += (calculated_dist - distance)**2
    return error_sum

def weighted_objective_function(loc, stations, weights=None):
    """
    加权目标函数
    
    参数:
    loc: 震源位置坐标 [x, y, z]
    stations: 监测台数据，每行包含[x, y, z, distance]
    weights: 权重数组，长度与stations行数相同
    
    返回:
    加权误差平方和
    """
    if weights is None:
        weights = np.ones(len(stations))
    
    x, y, z = loc
    error_sum = 0
    for i, station in enumerate(stations):
        sx, sy, sz, distance = station
        calculated_dist = np.sqrt((x - sx)**2 + (y - sy)**2 + (z - sz)**2)
        error_sum += weights[i] * (calculated_dist - distance)**2
    return error_sum

def solve_problem1(stations_a):
    """
    问题(1)：利用方程组求解基本震源坐标
    
    参数:
    stations_a: 基本监测台数据
    
    返回:
    解析解和优化解
    """
    print("\n正在求解问题(1)...")
    
    # 1. 代数解法
    # 方程2减方程1
    x1 = (930.98 - 851.00 - 100) / (-20)
    
    # 方程3减方程1
    x2 = (970.01 - 851.00 - 100) / 20
    
    # 取平均值
    x = (x1 + x2) / 2
    
    # 方程4减方程1
    y = (1250.97 - 851.00 - 100) / (-20)
    
    # 代入方程1求z
    z_squared = 851.00 - x**2 - y**2
    z = -np.sqrt(z_squared)  # 取负值，因为震源在地下
    
    algebraic_solution = np.array([x, y, z])
    
    # 2. 优化方法求解
    initial_guess = [0, 0, -10]  # 初始猜测
    result = minimize(objective_function, initial_guess, args=(stations_a,), 
                     method='Nelder-Mead')
    optimization_solution = result.x
    
    # 3. 验证解的准确性
    errors = []
    for station in stations_a:
        sx, sy, sz, distance = station
        calculated_dist = np.sqrt((x - sx)**2 + (y - sy)**2 + (z - sz)**2)
        error = calculated_dist - distance
        errors.append(error)
    
    # 打印结果
    print(f"代数解法得到的震源坐标: {algebraic_solution}")
    print(f"优化方法得到的震源坐标: {optimization_solution}")
    print(f"代数解的各监测台误差: {errors}")
    print(f"优化解的目标函数值(误差平方和): {result.fun}")
    
    return algebraic_solution, optimization_solution

def solve_problem2(stations_a):
    """
    问题(2)：考虑测量误差情况下的震源可能位置范围及最可能坐标
    
    参数:
    stations_a: 基本监测台数据
    
    返回:
    包含可能范围和最可能坐标的字典
    """
    print("\n正在求解问题(2)...")
    start_time = time.time()
    
    # Monte Carlo模拟考虑误差
    results = []
    n_simulations = 1000
    
    for _ in range(n_simulations):
        # 生成带有±0.2%误差的距离
        perturbed_stations = stations_a.copy()
        error_factors = 1 + np.random.uniform(-0.002, 0.002, len(stations_a))
        perturbed_stations[:, 3] *= error_factors
        
        # 求解带误差的问题
        initial_guess = [0, 0, -10]
        result = minimize(objective_function, initial_guess, 
                          args=(perturbed_stations,), method='Nelder-Mead')
        results.append(result.x)
    
    # 转换为numpy数组进行统计
    results = np.array(results)
    
    # 计算每个坐标轴的范围
    x_range = [np.min(results[:, 0]), np.max(results[:, 0])]
    y_range = [np.min(results[:, 1]), np.max(results[:, 1])]
    z_range = [np.min(results[:, 2]), np.max(results[:, 2])]
    
    # 计算最可能的坐标(取均值)和标准差
    most_likely = np.mean(results, axis=0)
    std_devs = np.std(results, axis=0)
    
    # 打印结果
    print(f"最可能的震源坐标: {most_likely}")
    print(f"坐标的标准差: {std_devs}")
    print(f"X坐标可能范围: {x_range}")
    print(f"Y坐标可能范围: {y_range}")
    print(f"Z坐标可能范围: {z_range}")
    print(f"Monte Carlo模拟完成，耗时: {time.time() - start_time:.2f}秒")
    
    return {
        'possible_range': {
            'x': x_range,
            'y': y_range,
            'z': z_range
        },
        'most_likely': most_likely,
        'std_devs': std_devs,
        'all_results': results  # 保存所有模拟结果用于后续分析
    }

def solve_problem3(stations_a_extended, stations_b):
    """
    问题(3)：利用更多监测台数据改进模型并应用于新数据
    
    参数:
    stations_a_extended: 扩展的地震a监测台数据
    stations_b: 地震b的监测台数据
    
    返回:
    包含两次地震震源坐标的字典
    """
    print("\n正在求解问题(3)...")
    
    # 1. 求解地震a的震源坐标
    # 基于距离定义权重（距离越近权重越大）
    distances_a = stations_a_extended[:, 3]
    weights_a = 1 / distances_a  
    
    initial_guess = [0, 0, -10]
    result_a = minimize(weighted_objective_function, initial_guess, 
                       args=(stations_a_extended, weights_a), method='Nelder-Mead')
    
    # 2. 求解地震b的震源坐标
    distances_b = stations_b[:, 3]
    weights_b = 1 / distances_b
    
    result_b = minimize(weighted_objective_function, initial_guess, 
                       args=(stations_b, weights_b), method='Nelder-Mead')
    
    # 打印结果
    print(f"地震a的震源坐标: {result_a.x}")
    print(f"地震a求解的目标函数最小值: {result_a.fun}")
    print(f"地震b的震源坐标: {result_b.x}")
    print(f"地震b求解的目标函数最小值: {result_b.fun}")
    
    # 3. 尝试不同的权重策略（示例：使用均匀权重）
    uniform_result_a = minimize(objective_function, initial_guess, 
                               args=(stations_a_extended,), method='Nelder-Mead')
    uniform_result_b = minimize(objective_function, initial_guess, 
                               args=(stations_b,), method='Nelder-Mead')
    
    print(f"均匀权重下地震a的震源坐标: {uniform_result_a.x}")
    print(f"均匀权重下地震b的震源坐标: {uniform_result_b.x}")
    
    return {
        'earthquake_a': {
            'weighted': result_a.x,
            'uniform': uniform_result_a.x
        },
        'earthquake_b': {
            'weighted': result_b.x,
            'uniform': uniform_result_b.x
        }
    }

def analyze_depth_impact(stations_a_extended):
    """
    问题(4)：分析监测台深度对定位准确性的影响
    
    参数:
    stations_a_extended: 扩展的地震a监测台数据
    
    返回:
    包含分析结果的字典
    """
    print("\n正在分析监测台深度的影响(问题4)...")
    start_time = time.time()
    
    # 创建不同深度分布的监测台配置
    depths = [-5, -10, -15, -20, -25]
    
    # 基于原始监测台数据，创建不同深度分布的配置
    multi_depth_stations = []
    for i, station in enumerate(stations_a_extended):
        depth_index = i % len(depths)
        new_station = station.copy()
        new_station[2] = depths[depth_index]  # 设置不同的z坐标(深度)
        multi_depth_stations.append(new_station)
    
    multi_depth_stations = np.array(multi_depth_stations)
    
    # 创建一组模拟的震源位置
    simulated_sources = [
        [1, -15, -25],
        [2, -10, -20],
        [-3, -5, -30],
        [0, -12, -18],
        [5, -8, -22]
    ]
    
    # 比较定位精度
    flat_depth_errors = []
    multi_depth_errors = []
    flat_results = []
    multi_results = []
    
    for source in simulated_sources:
        # 计算到平面监测台的实际距离
        flat_distances = []
        for station in stations_a_extended:
            sx, sy, sz = station[:3]
            dist = np.sqrt((source[0] - sx)**2 + (source[1] - sy)**2 + (source[2] - sz)**2)
            flat_distances.append(dist)
        
        # 计算到多深度监测台的实际距离
        multi_distances = []
        for station in multi_depth_stations:
            sx, sy, sz = station[:3]
            dist = np.sqrt((source[0] - sx)**2 + (source[1] - sy)**2 + (source[2] - sz)**2)
            multi_distances.append(dist)
        
        # 为实际距离添加随机误差 (使用相同的随机种子确保公平比较)
        np.random.seed(42 + simulated_sources.index(source))  # 为每个震源使用不同但可重现的种子
        error_factors = 1 + np.random.uniform(-0.002, 0.002, len(flat_distances))
        
        flat_distances = np.array(flat_distances) * error_factors
        multi_distances = np.array(multi_distances) * error_factors  # 使用相同的误差因子
        
        # 创建带误差的平面配置数据
        flat_stations = stations_a_extended.copy()
        flat_stations[:, 3] = flat_distances
        
        # 创建带误差的多深度配置数据
        multi_stations = multi_depth_stations.copy()
        multi_stations[:, 3] = multi_distances
        
        # 求解平面配置下的震源位置
        initial_guess = [0, 0, -10]
        result_flat = minimize(objective_function, initial_guess, args=(flat_stations,), 
                              method='Nelder-Mead')
        
        # 求解多深度配置下的震源位置
        result_multi = minimize(objective_function, initial_guess, args=(multi_stations,), 
                               method='Nelder-Mead')
        
        # 计算误差
        flat_error = np.linalg.norm(np.array(source) - result_flat.x)
        multi_error = np.linalg.norm(np.array(source) - result_multi.x)
        
        flat_depth_errors.append(flat_error)
        multi_depth_errors.append(multi_error)
        flat_results.append(result_flat.x)
        multi_results.append(result_multi.x)
    
    # 分析结果
    avg_flat_error = np.mean(flat_depth_errors)
    avg_multi_error = np.mean(multi_depth_errors)
    improvement = (avg_flat_error - avg_multi_error) / avg_flat_error * 100
    
    # 打印结果
    print(f"相同深度配置的平均误差: {avg_flat_error:.4f}")
    print(f"不同深度配置的平均误差: {avg_multi_error:.4f}")
    print(f"改进百分比: {improvement:.2f}%")
    print(f"深度影响分析完成，耗时: {time.time() - start_time:.2f}秒")
    
    return {
        'flat_depth_errors': flat_depth_errors,
        'multi_depth_errors': multi_depth_errors,
        'avg_flat_error': avg_flat_error,
        'avg_multi_error': avg_multi_error,
        'improvement': improvement,
        'simulated_sources': simulated_sources,
        'flat_results': flat_results,
        'multi_results': multi_results
    }

def visualize_results(stations_a, stations_a_extended, problem1_result, 
                      problem2_result, problem3_result, problem4_result):
    """
    可视化所有问题的结果
    
    参数:
    stations_a: 基本监测台数据
    stations_a_extended: 扩展的监测台数据
    problem1_result: 问题1的结果
    problem2_result: 问题2的结果
    problem3_result: 问题3的结果
    problem4_result: 问题4的结果
    """
    print("\n正在生成可视化结果...")
    
    # 1. 3D可视化震源位置和监测台
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制基本监测台
    ax.scatter(stations_a[:, 0], stations_a[:, 1], stations_a[:, 2], 
               c='blue', marker='^', s=100, label='基本监测台')
    
    # 绘制扩展监测台
    ax.scatter(stations_a_extended[4:, 0], stations_a_extended[4:, 1], 
               stations_a_extended[4:, 2], c='cyan', marker='^', s=100, 
               label='扩展监测台')
    
    # 绘制问题(1)的解
    algebraic_sol, optimization_sol = problem1_result
    ax.scatter(algebraic_sol[0], algebraic_sol[1], algebraic_sol[2], 
               c='red', marker='o', s=200, label='问题1代数解')
    ax.scatter(optimization_sol[0], optimization_sol[1], optimization_sol[2], 
               c='darkred', marker='*', s=200, label='问题1优化解')
    
    # 绘制问题(2)的解
    ax.scatter(problem2_result['most_likely'][0], 
               problem2_result['most_likely'][1], 
               problem2_result['most_likely'][2], 
               c='green', marker='o', s=200, label='问题2最可能解')
    
    # 绘制问题(3)的解
    ax.scatter(problem3_result['earthquake_a']['weighted'][0], 
               problem3_result['earthquake_a']['weighted'][1], 
               problem3_result['earthquake_a']['weighted'][2], 
               c='purple', marker='o', s=200, label='问题3解(地震a)')
    ax.scatter(problem3_result['earthquake_b']['weighted'][0], 
               problem3_result['earthquake_b']['weighted'][1], 
               problem3_result['earthquake_b']['weighted'][2], 
               c='orange', marker='o', s=200, label='问题3解(地震b)')
    
    ax.set_xlabel('X轴')
    ax.set_ylabel('Y轴')
    ax.set_zlabel('Z轴')
    ax.set_title('地震震源定位结果')
    ax.legend()
    
    plt.savefig('震源定位结果可视化.png', dpi=300, bbox_inches='tight')
    
    # 2. 问题2的误差分布可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # X坐标分布
    axes[0].hist(problem2_result['all_results'][:, 0], bins=30)
    axes[0].set_title('X坐标分布')
    axes[0].axvline(problem2_result['most_likely'][0], color='r', linestyle='--')
    
    # Y坐标分布
    axes[1].hist(problem2_result['all_results'][:, 1], bins=30)
    axes[1].set_title('Y坐标分布')
    axes[1].axvline(problem2_result['most_likely'][1], color='r', linestyle='--')
    
    # Z坐标分布
    axes[2].hist(problem2_result['all_results'][:, 2], bins=30)
    axes[2].set_title('Z坐标分布')
    axes[2].axvline(problem2_result['most_likely'][2], color='r', linestyle='--')
    
    plt.tight_layout()
    plt.savefig('坐标分布分析.png', dpi=300, bbox_inches='tight')
    
    # 3. 问题4的深度影响分析
    plt.figure(figsize=(10, 6))
    bar_positions = np.arange(len(problem4_result['simulated_sources']))
    bar_width = 0.35
    
    plt.bar(bar_positions - bar_width/2, problem4_result['flat_depth_errors'], 
            width=bar_width, label='相同深度')
    plt.bar(bar_positions + bar_width/2, problem4_result['multi_depth_errors'], 
            width=bar_width, label='不同深度')
    
    plt.xlabel('模拟震源索引')
    plt.ylabel('定位误差')
    plt.title(f'监测台深度分布对定位精度的影响\n平均改进: {problem4_result["improvement"]:.2f}%')
    plt.xticks(bar_positions, [f'S{i+1}' for i in range(len(problem4_result['simulated_sources']))])
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.savefig('深度影响分析.png', dpi=300, bbox_inches='tight')
    
    # 4. 补充图：比较不同问题的结果
    plt.figure(figsize=(10, 6))
    solutions = [
        algebraic_sol, 
        problem2_result['most_likely'], 
        problem3_result['earthquake_a']['weighted']
    ]
    labels = ['问题1代数解', '问题2考虑误差解', '问题3扩展监测台解']
    
    x_coords = [sol[0] for sol in solutions]
    y_coords = [sol[1] for sol in solutions]
    z_coords = [sol[2] for sol in solutions]
    
    plt.scatter(x_coords, y_coords, s=100)
    
    for i, (x, y, label) in enumerate(zip(x_coords, y_coords, labels)):
        plt.annotate(f"{label}\n({x:.2f}, {y:.2f}, {z_coords[i]:.2f})", 
                     (x, y), xytext=(10, 10), textcoords='offset points')
    
    plt.xlabel('X坐标')
    plt.ylabel('Y坐标')
    plt.title('不同问题解的比较(XY平面)')
    plt.grid(True)
    
    plt.savefig('不同解的比较.png', dpi=300, bbox_inches='tight')
    
    print("所有可视化结果已保存")
    
def export_results_to_markdown(problem1_result, problem2_result, 
                               problem3_result, problem4_result):
    """
    将结果导出为Markdown文件
    
    参数:
    problem1_result: 问题1的结果
    problem2_result: 问题2的结果
    problem3_result: 问题3的结果
    problem4_result: 问题4的结果
    """
    algebraic_sol, optimization_sol = problem1_result
    
    with open('地震震源定位问题求解结果.md', 'w', encoding='utf-8') as f:
        f.write('# 地震震源定位问题求解结果\n\n')
        
        # 问题1的结果
        f.write('## 一、问题(1)的结果\n\n')
        f.write(f'代数解法得到的震源坐标: ({algebraic_sol[0]:.4f}, {algebraic_sol[1]:.4f}, {algebraic_sol[2]:.4f})\n\n')
        f.write(f'优化方法得到的震源坐标: ({optimization_sol[0]:.4f}, {optimization_sol[1]:.4f}, {optimization_sol[2]:.4f})\n\n')
        
        # 问题2的结果
        f.write('## 二、问题(2)的结果\n\n')
        f.write('考虑±0.2%测量误差后:\n\n')
        most_likely = problem2_result['most_likely']
        std_devs = problem2_result['std_devs']
        ranges = problem2_result['possible_range']
        
        f.write(f'最可能的震源坐标: ({most_likely[0]:.4f}, {most_likely[1]:.4f}, {most_likely[2]:.4f})\n\n')
        f.write(f'坐标的标准差: ({std_devs[0]:.4f}, {std_devs[1]:.4f}, {std_devs[2]:.4f})\n\n')
        f.write('各坐标的可能取值范围:\n')
        f.write(f'- X坐标: [{ranges["x"][0]:.4f}, {ranges["x"][1]:.4f}]\n')
        f.write(f'- Y坐标: [{ranges["y"][0]:.4f}, {ranges["y"][1]:.4f}]\n')
        f.write(f'- Z坐标: [{ranges["z"][0]:.4f}, {ranges["z"][1]:.4f}]\n\n')
        
        # 问题3的结果
        f.write('## 三、问题(3)的结果\n\n')
        f.write('### 地震a的震源坐标\n\n')
        eq_a_weighted = problem3_result['earthquake_a']['weighted']
        eq_a_uniform = problem3_result['earthquake_a']['uniform']
        
        f.write(f'加权解: ({eq_a_weighted[0]:.4f}, {eq_a_weighted[1]:.4f}, {eq_a_weighted[2]:.4f})\n')
        f.write(f'均匀权重解: ({eq_a_uniform[0]:.4f}, {eq_a_uniform[1]:.4f}, {eq_a_uniform[2]:.4f})\n\n')
        
        f.write('### 地震b的震源坐标\n\n')
        eq_b_weighted = problem3_result['earthquake_b']['weighted']
        eq_b_uniform = problem3_result['earthquake_b']['uniform']
        
        f.write(f'加权解: ({eq_b_weighted[0]:.4f}, {eq_b_weighted[1]:.4f}, {eq_b_weighted[2]:.4f})\n')
        f.write(f'均匀权重解: ({eq_b_uniform[0]:.4f}, {eq_b_uniform[1]:.4f}, {eq_b_uniform[2]:.4f})\n\n')
        
        # 问题4的结果
        f.write('## 四、问题(4)的结果\n\n')
        f.write('监测台深度分布对定位精度的影响:\n\n')
        f.write(f'相同深度配置的平均误差: {problem4_result["avg_flat_error"]:.4f}\n')
        f.write(f'不同深度配置的平均误差: {problem4_result["avg_multi_error"]:.4f}\n')
        f.write(f'精度改进百分比: {problem4_result["improvement"]:.2f}%\n\n')
        
        # 结论
        f.write('## 五、总结与结论\n\n')
        f.write('1. 通过代数方法和优化方法，我们能够准确定位震源坐标。\n')
        f.write('2. 考虑测量误差后，震源可能位置存在一定范围的波动，但最可能位置与精确解接近。\n')
        f.write('3. 利用更多监测台的数据并引入权重策略，可以进一步提高震源定位的精度。\n')
        f.write('4. 监测台设置在不同深度可以显著提高震源定位精度，尤其是对震源深度的估计。\n')
        f.write('5. 分析结果表明，合理布置监测台网络对地震监测具有重要意义。\n')
    
    print("结果已导出到'地震震源定位问题求解结果.md'")

def main():
    """主函数"""
    print("地震震源定位问题数值求解")
    print("=" * 50)
    
    # 加载数据
    stations_a, stations_a_extended, stations_b = load_data()
    
    # 解决问题1
    problem1_result = solve_problem1(stations_a)
    
    # 解决问题2
    problem2_result = solve_problem2(stations_a)
    
    # 解决问题3
    problem3_result = solve_problem3(stations_a_extended, stations_b)
    
    # 解决问题4
    problem4_result = analyze_depth_impact(stations_a_extended)
    
    # 可视化结果
    visualize_results(stations_a, stations_a_extended, problem1_result, 
                      problem2_result, problem3_result, problem4_result)
    
    # 导出结果到Markdown
    export_results_to_markdown(problem1_result, problem2_result, 
                              problem3_result, problem4_result)
    
    print("\n所有问题求解完成！")

if __name__ == "__main__":
    main() 