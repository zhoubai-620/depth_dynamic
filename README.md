# Extreme Avoid: Dynamic Obstacle Tracking & Avoidance System

极端环境下的动态障碍物追踪与避障系统。融合 **DepthNav** (BPTT导航框架) 与 **DPTracker** (双提示夜间目标跟踪) 两个项目的核心能力。

## 三大创新点

| 创新点 | 描述 | 核心模块 |
|--------|------|----------|
| **创新点一** | 双提示感知主干接入DepthNav策略网络 | `perception/` — FusedBackbone + TrackerFusedExtractor |
| **创新点二** | ToA场 → 动态碰撞风险场 | `risk/ttc_field.py` — 可微分TTC解析风险场 |
| **创新点三** | 导航效率偏航 → 跟踪置信度偏航 | `risk/confidence_proxy.py` — 几何代理置信度 |

## 目录结构

```
code/
├── extreme_avoid/          # 主包
│   ├── perception/         # 感知层 (创新点一)
│   │   ├── fused_backbone.py
│   │   ├── obstacle_head.py
│   │   └── tracker_fused_extractor.py
│   ├── prediction/         # 运动预测 (创新点二)
│   │   └── motion_head.py
│   ├── risk/               # 风险场 & 置信度代理 (创新点二、三)
│   │   ├── ttc_field.py
│   │   └── confidence_proxy.py
│   ├── envs/               # 环境层
│   │   ├── dynamic_obstacle_manager.py
│   │   └── dynamic_avoidance_env.py
│   ├── policies/           # 策略层
│   │   └── tracker_fused_policy.py
│   ├── data/               # 离线数据采集与拟合
│   │   ├── collect_confidence_logs.py
│   │   └── fit_confidence_proxy.py
│   ├── scripts/            # 训练/评估/场景生成
│   │   ├── train_bptt_ext.py
│   │   ├── generate_dynamic_scenes.py
│   │   └── eval_avoidance.py
│   ├── configs/            # YAML配置文件
│   │   ├── policy_cfg/
│   │   ├── train_cfg/      # stage1~4 课程学习
│   │   └── eval_cfg/
│   └── registry.py         # 注册表扩展
├── tests/                  # 测试套件
├── requirements.txt
└── setup.py
```

## 课程学习流程

按 skill.md §6 实施顺序：

1. **Stage 1** (`stage1_static.yaml`): 静态环境baseline，用原生 NavigationEnv
2. **Stage 2** (`stage2_dynamic_gt.yaml`): 动态障碍物 + **GT真值** → 验证风险场设计
3. **Stage 3** (`stage3_perception.yaml`): 接入真实感知 (TrackerFusedExtractor)，对比stage2差异
4. **Stage 4** (`stage4_extreme.yaml`): 全流程 + 光照退化 + 多障碍物

## 关键设计决策

- **不可导渲染绕过**: habitat-sim的渲染管线不可导 → 用 ConfidenceProxy 几何代理函数替代
- **显存优化**: BPTT horizon步长累积计算图 → prompters做了结构性瘦身 (降维、减层)
- **真值隔离**: DynamicObstacleManager的GT数据 vs 感知估计值 → 两条独立数据流

## 依赖

- DepthNav: `depthnav/` (rislab/depthnav)
- DPTracker: `DPTracker/` (yiheng-wang-duke/DPTracker)
- habitat-sim (build from source)
- PyTorch 2.2.x + CUDA
