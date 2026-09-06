# VERITAS — ICDM26 复现工程

Trust-But-Verify: Poisoning-Resilient Locally Private Graph Learning Protocols
本文档对应对论文中 **VERITAS** 主方法(from-scratch re-implementation)的复现。
工程从论文算法描述从零实现(论文脚注:官方代码 acceptance 后公开,本项目为独立复现)。

## 1. 论文方法回顾(VERITAS 四阶段)

| 阶段 | 论文位置 | 本工程实现 |
|---|---|---|
| ① Local Data Perturbation | Alg.2、Eq.(10)-(11) | `veritas/mechanisms.py`(特征)+ `veritas/structure.py`(verification list + KRR) |
| ② Attestation-Driven Pruning | Eq.(13)-(16), Prop.2 | `veritas/structure.py::server_prune` |
| ③ Utility Restoration(双去噪) | Sect. IV-C | `veritas/denoise.py`(NFR + HOA) |
| ④ Robust Private Graph Learning | Sect. IV-D | `veritas/models.py` 两层 GCN(hidden=64) |

### ① 用户端扰动
- **节点特征**(Alg. 2):`m = clamp(⌊δ·ϵx⌋,1,d)`,均匀无放回采样 m 个维度,每个采样维度用 `(ϵx/m)`-LDP 的一维机制独立扰动,其余维度置 0,随后 RECT 无偏校正。
  支持 6 种机制:`PM`(论文 Eq.(4)-(5),精确实现)、`MB`(CCS'21 LPGNN 风格多比特核)、`SW`(SIGMOD'20 square wave 风格)、`LP`、`AG`(解析高斯)、`1B`(one-bit)。
- **图结构(verification list)**:每个节点 v 对其邻接维护 `ă_v ∈ {0,…,κ}^|V|`(Eq.10),κ=5;每条有向声称独立做 (κ+1)-元 KRR(Eq.11),预算 ϵa=5.0。

### ② 服务端剪枝
对每个候选对 (u,v) 提取双边信号并计算:
- `s_exist = 1[ã_uv≠0]·1[ã_vu≠0]`(存在门)
- `s_cons = min(ã_uv, ã_vu)`(一致性,min 抵消恶意单方面报高)
- `s_asym = |ã_uv − ã_vu|`(不对称惩罚)
- 边置信 `c_uv = s_exist·(s_cons/κ)·exp(−λ·s_asym)`(Eq.13, λ=1.0)
- 节点嫌疑 `ϕ_v = mean_u(ã_vu − ã_uv)`(Eq.14 的语义:恶意节点给他人信任 > 他人给它的信任)→ 剪除 `{v: ϕ_v > γ}`(Eq.16)。默认 γ 取 Prop.2 理论期望中点,并设置最小声称数下限避免误剪低度良性节点。

### ③ 双去噪(NFR + HOA)
论文未公开 NFR/HOA 的精确公式(引自已发表的前作)。本实现采用如下等价近似并在复现中保持其"在服务端后处理、不消耗额外隐私预算"的性质:
- NFR:对带噪特征做稳健(中位数)中心化——扰动后特征在 Alg.2 下高度稀疏,该算子对纯噪声/未观测坐标恒等,不会像硬性维度清零那样破坏学习;
- HOA:个性化多跳聚合 `X ← (1−α)X + α·D⁻¹ÂX`(hops=1, α=0.5),通过邻居平均降低 LDP 噪声方差。

> ⚠️ 从零复现的不确定点(论文未公开细节)汇总见文末「复现假设」。

## 2. 数据集与实验设置(论文 Sect. V)

| 数据集 | 类型 | 来源 | 节点/边/特征/类 | 本次复现 |
|---|---|---|---|---|
| Cora | citation | PyG Planetoid | 2708 / 5278 / 1433 / 7 | ✅ |
| Citeseer | citation | PyG Planetoid | 3327 / 4552 / 3703 / 6 | ✅ |
| LastFM | social | PyG LastFMAsia | 7624 / 27806 / 7842 / 18 | ⛔ 官方源 graphmining.ai 已下线 |
| Twitch | social | PyG Twitch(ES) | 4648 / 61706 / 128 / 2 | ⛔ 官方源 graphmining.ai 已下线 |

> LastFM 与 Twitch 的官方数据托管(graphmining.ai)当前不可达,本机与服务器均无法
> 下载,也未找到可用镜像(见下)。因此本次实验在 **Cora + Citeseer** 上完成;
> 一旦获得这两个数据集文件(放入 `data/raw/lastfm_asia.npz` 与
> `data/ES/raw/ES.json`),直接运行 `run_fig3.py` 即可补全全部四个数据集。

设置:随机 50% / 25% / 25% 划分;默认 ϵx=0.1、ϵa=5.0、κ=5、注入率 r⋆=0.01;结果 = 10 次独立运行平均 ± 95% CI(与论文一致)。数据特征逐维缩放到 [−1,1]。

## 3. 工程结构

```
ICDM26/
├── veritas/
│   ├── mechanisms.py   # 6 种一维 LDP 机制 + Alg.2 特征扰动管线
│   ├── structure.py    # verification list 构造、KRR、Stage-② 双边剪枝
│   ├── attack.py       # Alg.1 注入攻击(黑盒威胁模型)
│   ├── denoise.py      # Stage-③ NFR/HOA
│   ├── models.py       # Stage-④ 两层 GCN
│   ├── data.py         # 数据集加载 + 预处理 + 划分
│   ├── pipeline.py     # Algorithm 3 端到端流程
│   └── main.py         # 单配置运行入口
├── run_fig3.py         # 批量跑 4 数据集 × 6 机制 × 10 seeds(VERITAS)
├── summarize_fig3.py   # 汇总 mean ± 95% CI,输出表格/图
├── requirements.txt
└── README.md
```

## 4. 运行方式

```bash
# 单配置
python -m veritas.main --dataset Cora --mech PM --defense veritas --seed 0

# 全量(默认只跑 VERITAS 主方法:4 数据集 × 6 机制 × 10 seeds = 240 runs)
python run_fig3.py --workers 6 --seeds 10

# 汇总
python summarize_fig3.py        # -> results/fig3.csv, results/fig3_summary.npy, results/fig3.png
```

命令行常用参数:`--mech {1B,LP,AG,SW,MB,PM}`、`--defense veritas|baseline`、
`--eps-x 0.1 --eps-a 5.0 --r-star 0.01 --t 12 --attack-style mean --gamma`。

## 5. 复现假设(from-scratch,README 记录)

1. 特征域:数据每维 `max|x|` 归一化到 [−1,1](作为公开预处理)。
2. Alg.2 的 `δ`(m=⌊δ·ϵ⌋)论文未给数值,取 δ=30(ϵx=0.1 时每节点扰动 m=3 维)。
3. 良性节点间 verification-list 等级取 κ;良性 victim 对攻击者(社会工程好友)按 Prop.2 情景给最低正等级 1,攻击者对所有人报 κ。
4. 攻击者完全控制其"扰动后报告"(Alg.1 返回 reports):特征直接在 Range(M) 内取值(默认 'mean' 风格=其受害者原始特征均值),结构声称确定性地报 κ,不经过随机机制——这是 LDP 下开放参与威胁模型允许的最强构造。
5. NFR/HOA 采用近似实现(见上),NFR 对稀疏特征近似恒等,HOA 为主要去噪贡献。
6. GNN:2 层 GCN hidden 64,Adam(lr=1e-2, wd=5e-4),400 epochs 早停 patience=100;训练前对特征做逐维 z-score(服务端后处理)。
7. 攻击注入参数:t=12(每个恶意节点连接的 victim 数)、ff_degree=3(内部协调)。

## 6. 结果(本次复现,10 次独立运行,mean ± 95% CI)

攻击下 VERITAS 在 Cora / Citeseer 上的节点分类精度(默认 ϵx=0.1, ϵa=5.0,
κ=5, r⋆=0.01,平均 6 种机制):

| 数据集 | 1B | LP | AG | SW | MB | PM |
|---|---|---|---|---|---|---|
| Cora | 75.0±1.0 | 75.2±1.1 | 75.4±1.2 | 75.0±1.3 | 75.2±1.3 | 75.2±1.0 |
| Citeseer | 60.7±1.0 | 60.8±0.8 | 60.3±0.8 | 61.0±0.9 | 61.2±0.8 | 60.8±0.8 |

其中恶意节点(注入比例 1%)检出率 det_fake ≈ 93–100%,良性节点误剪率 ≈ 0%。
`results/fig3.csv` 为逐 seed 原始记录;`results/fig3_veritas.png` 为汇总图;
`python summarize_fig3.py` 可随时重新生成。完整数值表见 `results/`。
