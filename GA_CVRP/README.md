# GA_CVRP - 遗传算法求解容量约束车辆路径问题

## 问题描述

**容量约束车辆路径问题 (Capacitated Vehicle Routing Problem, CVRP)**

CVRP是VRP的基本形式，考虑车辆容量限制。目标是设计一组车辆路径，从配送中心出发，服务所有客户需求后返回配送中心，使得总运输成本最小。

### 数学模型

目标函数：
$min \sum_{k=1}^{K} \sum_{i=0}^{n} \sum_{j=0}^{n} d_{ij} x_{ijk}$

约束条件：
$\sum_{k=1}^{K} \sum_{i=0}^{n} x_{ijk} = 1, \quad \forall j = 1,2,...,n$

$\sum_{j=1}^{n} q_j \sum_{i=0}^{n} x_{ijk} \leq Q, \quad \forall k = 1,2,...,K$  (容量约束)

$\sum_{i=0}^{n} x_{ihk} - \sum_{j=0}^{n} x_{hjk} = 0, \quad \forall h = 1,2,...,n; \forall k = 1,2,...,K$

$\sum_{i=1}^{n} x_{i0k} = 1, \quad \sum_{j=1}^{n} x_{0jk} = 1, \quad \forall k = 1,2,...,K$

## 算法流程

### 基于遗传算法的CVRP求解

1. **编码方案**
   - 使用路径表示法：将VRP解表示为包含配送中心的长序列
   - 编码示例：[1,3,5,1,2,4,1]表示两辆车：路线1：0→3→5→0，路线2：0→2→4→0

2. **初始化种群**
   - 随机生成满足容量约束的初始解
   - 使用贪心算法生成部分可行解以提高初始质量

3. **适应度评估**
   - 计算每条路径的总距离
   - 检查容量约束：对违反约束的解进行惩罚
   - 适应度函数：$f(x) = 1 / (total\_distance + penalty)$

4. **遗传操作**
   - **选择**：基于适应度的轮盘赌选择
   - **交叉**：路径交叉，保持子路径结构
   - **变异**：交换、插入、逆转等操作，确保容量约束

## 算法逻辑框架

### GA-CVRP完整优化流程图
```mermaid
graph TD
    A[开始] --> B[初始化GA参数]
    B --> C[加载CVRP数据]
    C --> D[城市坐标]
    C --> E[客户需求量]
    C --> F[车辆容量]
    C --> G[距离矩阵]
    D --> H[设置种群大小]
    H --> I[种群大小 NIND = 60]
    I --> J[最大代数 MAXGEN = 100]
    J --> K[初始化可行种群]
    K --> L[生成满足容量约束的初始解]
    L --> M[计算初始适应度]
    M --> N[记录最优个体]
    N --> O{达到最大代数?}
    O -->|否| P[主遗传循环]
    P --> Q[选择操作阶段]
    Q --> R[基于适应度轮盘赌选择]
    R --> S[交叉操作阶段]
    S --> T[交叉概率 Pc = 0.9]
    T --> U{执行交叉?}
    U -->|是| V[路径交叉操作]
    V --> W[保持子路径结构]
    U -->|否| X[跳过交叉]
    W --> Y[变异操作阶段]
    X --> Y
    Y --> Z[变异概率 Pm = 0.05]
    Z --> AA{执行变异?}
    AA -->|是| AB[交换/插入/逆转变异]
    AA -->|否| AC[跳过变异]
    AB --> AD[修复操作阶段]
    AC --> AD
    AD --> AE[容量约束检查]
    AE --> AF{容量超载?}
    AF -->|是| AG[修复不可行解]
    AF -->|否| AH[解有效]
    AG --> AI[重新分配超载客户]
    AI --> AJ[验证新分配]
    AJ --> AH
    AH --> AK[计算新适应度]
    AK --> AL[精英保留策略]
    AL --> AM[保留最优个体]
    AM --> AN[更新群体]
    AN --> AO[记录最优解]
    AO --> O
    O -->|是| AP[输出最优CVRP方案]
    AP --> AQ[最优车辆路径]
    AQ --> AR[总运输距离]
    AR --> AS[车辆使用数量]
    AS --> AT[每车载货量]
    AT --> AU[可视化结果]
    AU --> AV[结束]
```

### 容量约束修复机制详解
```mermaid
graph TD
    A[染色体输入] --> B[解析路径序列]
    B --> C[识别配送中心分隔符]
    C --> D[分割车辆子路径]
    D --> E[逐路径容量检查]
    E --> F[计算子路径总需求]
    F --> G{需求 ≤ 容量?}
    G -->|是| H[路径有效]
    G -->|否| I[检测到超载]
    I --> J[计算超载量]
    J --> K[识别超载客户]
    K --> L[选择修复策略]
    L --> M[策略1: 客户重分配]
    L --> N[策略2: 新建子路径]
    L --> O[策略3: 路径分割]
    
    M --> P[找到最近可行路径]
    N --> Q[创建新车辆路径]
    O --> R[将超载客户分组]
    
    P --> S[重新插入客户]
    Q --> T[分配超载客户到新路径]
    R --> U[分割为多个可行路径]
    
    S --> V[验证新路径]
    T --> V
    U --> V
    V --> W{修复成功?}
    W -->|是| X[返回可行染色体]
    W -->|否| Y[尝试其他策略]
    Y --> L
    H --> X
```

### 遗传操作详解
```mermaid
graph TD
    subgraph "父代选择"
        A[种群评估] --> B[计算适应度]
        B --> C[基于适应度排序]
        C --> D[轮盘赌选择父代]
    end
    
    subgraph "交叉操作"
        D --> E[选择交叉点]
        E --> F[保持配送中心结构]
        F --> G[交换客户序列]
        G --> H[生成子代染色体]
    end
    
    subgraph "变异操作"
        H --> I{选择变异类型}
        I -->|交换| J[交换两个客户位置]
        I -->|插入| K[移动客户到新位置]
        I -->|逆转| L[逆转子序列]
        J --> M[保持容量约束]
        K --> M
        L --> M
    end
    
    subgraph "修复验证"
        M --> N[容量约束检查]
        N --> O{验证通过?}
        O -->|是| P[接受新染色体]
        O -->|否| Q[应用修复机制]
        Q --> P
    end
```

### 路径编码与解码机制
```mermaid
graph TD
    subgraph "染色体编码"
        A[完整解] --> B[长序列表示]
        B --> C[1,3,5,2,1,4,6,1]
        C --> D[1 = 配送中心]
        D --> E[数字 = 客户编号]
    end
    
    subgraph "解码过程"
        F[染色体序列] --> G[按1分割]
        G --> H[提取子路径]
        H --> I[子路径1: 0→3→5→2→0]
        H --> J[子路径2: 0→4→6→0]
    end
    
    subgraph "实际路径"
        K[车辆1] --> L[路线: 0→3→5→2→0]
        M[车辆2] --> N[路线: 0→4→6→0]
    end
    
    subgraph "容量验证"
        L --> O[检查总需求]
        N --> P[检查总需求]
        O --> Q{≤ 容量?}
        P --> R{≤ 容量?}
        Q --> S[路径有效]
        R --> T[路径有效]
    end
```

### 适应度计算与评估
```mermaid
graph TD
    A[染色体输入] --> B[解码路径序列]
    B --> C[分割车辆子路径]
    C --> D[计算每条路径距离]
    D --> E[累加总距离]
    E --> F[容量约束检查]
    F --> G{容量满足?}
    G -->|是| H[适应度 = 1/总距离]
    G -->|否| I[适应度 = 极小值]
    I --> J[惩罚不可行解]
    H --> K[返回适应度值]
    J --> K
    K --> L[种群排序]
```

### 种群进化过程
```mermaid
graph LR
    subgraph "代际变化"
        A[初始随机种群] --> B[早期快速收敛]
        B --> C[中期精细搜索]
        C --> D[后期局部优化]
        D --> E[全局最优收敛]
    end
    
    subgraph "多样性维护"
        F[高多样性] --> G[逐步降低]
        G --> H[保持必要多样性]
        H --> I[防止过早收敛]
    end
    
    subgraph "精英策略"
        J[保留最优个体] --> K[确保不丢失]
        K --> L[引导种群进化]
        L --> M[收敛到最优解]
    end
```

### 伪代码框架
```
初始化GA-CVRP参数:
    种群大小 NIND = 60
    最大代数 MAXGEN = 100
    交叉概率 Pc = 0.9
    变异概率 Pm = 0.05
    代沟概率 GGAP = 0.9
    车辆容量 Q
    客户数量 N

加载CVRP数据:
    城市坐标 City.mat
    客户需求 Demands.mat
    车辆容量 Capacity.mat
    距离矩阵 Distance.mat

主优化过程:
% 初始化可行种群
种群 = InitPop(NIND, N, Demand, Capacity)
最优解 = []
最优适应度 = 0

for gen = 1 to MAXGEN:
    % 适应度评估
    [总距离, 适应度] = Fitness(种群, Distance, Demand, Capacity)
    
    % 记录最优个体
    [当前最优, 当前适应度] = findBest(种群, 适应度)
    if 当前适应度 > 最优适应度:
        最优解 = 当前最优
        最优适应度 = 当前适应度
    
    % 选择操作
    父代 = Select(种群, 适应度, GGAP)
    
    % 交叉操作
    子代 = []
    for i = 1 to size(父代,1)/2:
        if rand() < Pc:
            [子代1, 子代2] = Crossover(父代[i], 父代[i+1])
            子代.add(子代1)
            子代.add(子代2)
        else:
            子代.add(父代[i])
            子代.add(父代[i+1])
    
    % 变异操作
    for i = 1 to size(子代,1):
        if rand() < Pm:
            子代[i] = Mutate(子代[i])
    
    % 修复操作
    for i = 1 to size(子代,1):
        if not checkCapacity(子代[i], Demand, Capacity):
            子代[i] = repairSolution(子代[i], Demand, Capacity)
    
    % 精英保留
    子代 = Reins(子代, 最优解)
    
    % 更新种群
    种群 = 子代
    
    % 记录统计信息
    recordStatistics(gen, 最优适应度, 最优解)

输出最优CVRP方案:
    最优车辆路径 = 最优解
    总运输距离 = calculateTotalDistance(最优解, Distance)
    车辆使用数量 = countVehicles(最优解)
    每车载货量 = calculateLoadPerVehicle(最优解, Demand)
    
可视化CVRP路径
```

## 关键实现特点

### 容量约束处理
```matlab
% 路径合法性检查
function isValid = checkCapacity(route, Demand, Capacity)
    currentLoad = 0;
    for i = 1:length(route)
        if route(i) == 1  % 配送中心
            currentLoad = 0;  % 重置载货量
        else
            currentLoad = currentLoad + Demand(route(i));
            if currentLoad > Capacity
                isValid = false;
                return;
            end
        end
    end
    isValid = true;
end
```

### 路径分割机制
- 使用配送中心的出现来分割不同车辆的路径
- 每辆车从配送中心出发，服务若干客户后返回配送中心
- 确保每辆车的载货量不超过容量限制

### 适应度计算
```matlab
function [ttlDistance, FitnV] = Fitness(Chrom, Distance, Demand, Capacity)
    for i = 1:size(Chrom, 1)
        route = Chrom(i, :);
        totalDist = 0;
        currentLoad = 0;
        isValid = true;
        
        % 计算距离并检查容量约束
        for j = 2:length(route)
            totalDist = totalDist + Distance(route(j-1), route(j));
            if route(j) ~= 1
                currentLoad = currentLoad + Demand(route(j));
                if currentLoad > Capacity
                    totalDist = inf;  % 惩罚不可行解
                    isValid = false;
                    break;
                end
            else
                currentLoad = 0;  % 返回配送中心
            end
        end
        
        ttlDistance(i) = totalDist;
    end
    FitnV = 1 ./ (ttlDistance + eps);
end
```

## 文件结构

- `Main.m`：GA_CVRP主程序
- `InitPop.m`：初始化种群（生成可行解）
- `Fitness.m`：适应度计算（含约束检查）
- `Crossover.m`：交叉操作
- `Mutate.m`：变异操作
- `Select.m`：选择操作
- `Reins.m`：重插入操作
- `Reverse.m`：逆转操作
- `DrawPath.m`：路径可视化
- `TextOutput.m`：结果输出
- `dsxy2figxy.m`：坐标转换工具

## 参数配置

- 种群大小(NIND)：60
- 最大代数(MAXGEN)：100
- 交叉概率(Pc)：0.9
- 变异概率(Pm)：0.05
- 代沟概率(GGAP)：0.9

## 约束处理策略

### 1. 初始化约束
- 在初始化种群时确保每个解都满足容量约束
- 使用贪心算法或启发式方法生成初始可行解

### 2. 遗传操作约束
- 交叉和变异操作后检查新解的可行性
- 对不可行解进行修复或重新生成

### 3. 惩罚函数
- 对违反容量约束的解赋予极大适应度值
- 确保不可行解在进化过程中被淘汰

## 使用示例

1. 准备数据文件：
   - `City.mat`：客户和配送中心坐标
   - `Distance.mat`：距离矩阵
   - `Demand.mat`：各客户需求量
   - `Capacity.mat`：车辆容量

2. 运行`Main.m`执行求解

3. 输出结果：
   - 最优车辆路径方案
   - 每辆车的载货量
   - 总运输距离
   - 路径可视化图

## 性能优化

### 1. 初始解质量
- 使用节约算法(C-W算法)生成高质量初始解
- 结合最近邻算法提高初始种群质量

### 2. 局部搜索
- 在遗传算法中加入2-opt局部改进
- 提高解的局部优化能力

### 3. 自适应参数
- 根据收敛情况动态调整交叉和变异概率
- 平衡探索与开发能力

## 实际应用

CVRP广泛应用于：
- **物流配送**：最后一公里配送
- **供应链管理**：分销网络优化
- **零售配送**：连锁超市配送
- **电商物流**：快递配送路线规划
- **冷链运输**：温控产品配送