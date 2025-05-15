# 地震震源定位问题程序运行说明

## 一、程序概述

本程序`earthquake_source_localization.py`是针对地震震源定位问题的数学建模与求解实现，包含以下四个子问题的解决方案：

1. 基本震源坐标确定（利用4个监测台数据）
2. 考虑测量误差情况下的震源可能位置范围估计
3. 利用更多监测台数据改进模型并应用于新数据
4. 分析监测台埋于不同深度对定位准确性的影响

程序使用Python语言实现，采用数值优化和蒙特卡洛模拟等方法求解问题，并生成多种可视化结果。

## 二、环境配置

### 1. 必需的Python环境

- Python 3.6+（推荐使用Python 3.8或更高版本）
- 必需的Python库：
  - numpy：用于数值计算
  - scipy：用于优化求解
  - matplotlib：用于结果可视化
  - pandas（可选）：用于数据处理

### 2. 依赖库安装

可以使用以下命令安装所需的库：

```bash
pip install numpy scipy matplotlib pandas
```

如果使用Anaconda环境，可以使用：

```bash
conda install numpy scipy matplotlib pandas
```

## 三、运行方法

### 1. 直接运行方式

在命令行中，进入程序所在目录，执行以下命令：

```bash
python earthquake_source_localization.py
```

### 2. 通过IDE运行

如果使用PyCharm、VS Code等IDE，可以直接打开`earthquake_source_localization.py`文件并运行。

### 3. 运行过程说明

运行程序后，控制台将显示以下信息：

1. 程序开始执行的提示
2. 问题(1)的求解过程和结果
3. 问题(2)的求解过程和结果（包括蒙特卡洛模拟的进度）
4. 问题(3)的求解过程和结果
5. 问题(4)的分析过程和结果
6. 可视化结果生成的提示
7. 结果文件保存的提示

整个运行过程可能需要几分钟时间，特别是问题(2)和问题(4)中的蒙特卡洛模拟部分。

## 四、输出结果

程序执行完毕后，将生成以下输出文件：

### 1. 图形文件

- `震源定位结果可视化.png`：3D空间中显示各监测台位置和不同问题求得的震源位置
- `坐标分布分析.png`：问题(2)中考虑误差情况下震源坐标的分布直方图
- `深度影响分析.png`：问题(4)中不同深度配置对定位精度影响的对比图
- `不同解的比较.png`：不同问题求解结果的对比

### 2. 文本结果

- `地震震源定位问题求解结果.md`：包含所有问题的详细数值结果和分析

## 五、代码结构说明

程序代码按功能划分为以下几个主要部分：

1. **数据加载函数**：`load_data()`加载地震监测数据
2. **目标函数**：
   - `objective_function()`：基本的误差平方和目标函数
   - `weighted_objective_function()`：带权重的目标函数
3. **问题求解函数**：
   - `solve_problem1()`：求解问题(1)
   - `solve_problem2()`：求解问题(2)
   - `solve_problem3()`：求解问题(3)
   - `analyze_depth_impact()`：分析问题(4)
4. **可视化与结果导出函数**：
   - `visualize_results()`：生成图形结果
   - `export_results_to_markdown()`：导出文本结果
5. **主函数**：`main()`协调整个程序的执行流程

## 六、常见问题解答

### 1. 程序运行时间过长

- 可以通过减小问题(2)和问题(4)中的蒙特卡洛模拟次数来加快运行速度
- 在`solve_problem2()`函数中，将`n_simulations`的值从1000改为较小的值，如200

### 2. 图形显示问题

- 如果出现中文显示乱码，请确保系统安装了相应的中文字体
- 可以修改程序开头的`plt.rcParams['font.sans-serif']`设置

### 3. 结果精度问题

- 如需提高求解精度，可以修改`minimize()`函数的参数，如添加`tol`参数降低收敛容差

## 七、扩展与改进建议

如果需要扩展或改进程序，可以考虑以下方向：

1. 添加GUI界面，使程序操作更加直观
2. 实现地震波传播速度的变化模型，使模拟更加真实
3. 增加更多的优化算法比较，如遗传算法、粒子群算法等
4. 增加地震数据的导入导出功能，便于处理真实数据
5. 引入并行计算，加速蒙特卡洛模拟过程

## 八、参考资料

1. 最小二乘优化方法：[SciPy Optimization and Root Finding](https://docs.scipy.org/doc/scipy/reference/optimize.html)
2. 蒙特卡洛模拟：[Monte Carlo method](https://en.wikipedia.org/wiki/Monte_Carlo_method)
3. Nelder-Mead算法：[Nelder–Mead method](https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method) 