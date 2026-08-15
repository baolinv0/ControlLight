下面给出一份**可直接交给工程实现**的完整方案。目标不是训练模型，而是构建一个稳定的 **Reference-guided Aligned Pseudo-GT Selector**：

> 对每个输入样本，先生成一组与输入**像素严格对齐**的 brightness ladder 图像；
> 再以不对齐的生成式 pseudo-GT/reference 为风格参考，
> 从 ladder 中选出**人物影调风格最接近**的一张，作为最终用于训练的 **aligned pseudo-GT**。

---

# 1. 目标与设计原则

## 1.1 目标

给定：

* `source`：原始输入图像，后续训练时要求与伪 GT 像素级对齐；
* `reference`：由大模型增强得到的目标风格图像，但与 `source` **不严格对齐**，不可直接作为监督；
* `ladder = {I_k}`：由 `source` 生成的多个 brightness level 图像，全部与 `source` 严格对齐。

构建一个选择器，从 `ladder` 中选出：

[
I_{k^*} = \arg\min_k E(I_k,\ reference)
]

其中 `E` 不是全局亮度误差，而是 **人物中心的影调风格误差**。

最终输出：

* `aligned_pseudo_gt = I_{k^*}`
* 以及完整的候选可视化和打分结果。

---

## 1.2 为什么需要这个方案

当前问题是：

1. 生成式增强结果（reference/pseudo-GT）影调风格接近目标，但**几何、细节、纹理不稳定**；
2. 无法直接作为 supervised target；
3. 但其**人物亮度风格 + 人物/背景影调关系**仍然有价值；
4. brightness ladder 图像和 source 严格对齐，因此适合最终作为训练伪 GT。

所以本方案的本质是：

> **用生成式 pseudo-GT 提供风格指导，用 brightness ladder 提供几何对齐监督。**

---

# 2. 输入输出定义

## 2.1 输入

每个 sample 至少包含：

```text
sample_id
source_image
reference_image
ladder_images/
    a_m100.png
    a_m075.png
    a_m050.png
    a_m025.png
    a_000.png
    a_p025.png
    a_p050.png
    a_p075.png
    a_p100.png
```

### 说明

* `source_image` 与所有 `ladder_images` 必须严格对齐；
* `reference_image` 可不对齐；
* level 命名和 coefficient 固定。

---

## 2.2 输出

每个 sample 输出：

```text
output/
├── pseudo_gt/
│   └── {sample_id}.png
├── visualization/
│   └── {sample_id}.jpg
├── scores/
│   └── {sample_id}.json
└── debug/
    ├── masks/
    ├── overlays/
    └── crops/
```

---

# 3. 总体 Pipeline

```text
source
  ↓
generate ladder images (aligned)
  ↓
reference-guided level scoring
  ↓
best level selection
  ↓
aligned pseudo-GT
```

更细化为：

```text
source ------------------------------┐
                                     │
reference ----------------------┐    │
                                │    │
1) segmentation / region parse  │    │
   - person
   - face (optional)
   - local background ring      │    │
                                │    │
2) extract tone descriptors     │    │
   - reference descriptors      │    │
   - ladder descriptors         │    │
                                │    │
3) compute person-centric score │    │
                                │    │
4) rank all levels              │    │
                                │    │
5) select best level -----------┴----┘
                                ↓
                     aligned pseudo-GT
```

---

# 4. 核心思想：不要只按“亮度最近”选

不能只比较：

[
| \mu_{person}^{cand} - \mu_{person}^{ref} |
]

因为这会丢失：

* 人物内部 tone 分布；
* 人脸亮度；
* 人物和背景的明暗关系；
* 背景压暗程度；
* 高光/阴影层次。

真正需要匹配的是：

> **人物在场景中的影调位置，以及人物/背景之间的层次关系。**

因此评分需要升级为：

> **Person-centric Tone Descriptor Matching**

---

# 5. 区域定义

## 5.1 Person mask

### 目的

提取人物整体区域，用于人物整体影调分析。

### 实现建议

优先级：

1. 使用现有 human segmentation 模块；
2. 若已有 face/skin/human bundle，优先复用；
3. 若没有，推荐：

   * 检测：YOLO / RT-DETR / grounding person detector
   * 分割：SAM / SAM2 / SegFormer / person parsing

### 最终输出

```python
person_mask: H x W bool
```

要求：

* 包含主体人物；
* 支持多人场景时先选主人物（详见 §9）。

---

## 5.2 Face mask（可选但强烈建议）

### 目的

人脸通常是“人物影调风格”的核心感知区域。

### 实现建议

优先级：

1. 若已有 face parsing / face segmentation，直接使用；
2. 否则：

   * 人脸检测框；
   * 使用框内椭圆或 face parsing 获得粗人脸 mask。

### 输出

```python
face_mask: H x W bool or None
```

---

## 5.3 Local background mask

不能直接把 `~person_mask` 当背景，因为：

* 全局背景可能和人物无关；
* 远处天空/路面会干扰“人物与背景关系”。

建议使用 **人物周围的局部背景 ring**。

### 构造方法

```text
person mask
   ↓ dilate(r_outer)
outer region
   ↓ remove person
ring
   ↓ optional intersect with valid image area
local background
```

例如：

[
M_{bg} = \text{dilate}(M_{person}, r_{outer}) \setminus \text{dilate}(M_{person}, r_{inner})
]

建议：

* `r_inner = 5~15 px`
* `r_outer = 40~80 px`
* 或相对于 bbox 尺度：

  * `r_inner = 0.02 * max(h,w)`
  * `r_outer = 0.10 * max(h,w)`

### 输出

```python
local_bg_mask: H x W bool
```

---

# 6. 影调描述子设计

## 6.1 颜色空间

使用 **linear luminance**：

[
Y = 0.2126R + 0.7152G + 0.0722B
]

再进入 log domain：

[
L = \log_2(Y+\epsilon)
]

原因：

* 更接近 exposure / tone perception；
* 适合比较影调分布；
* 比 RGB mean 更稳定。

---

## 6.2 Person tone descriptor

对人物区域的 log-luminance 提取分位数：

[
T_P = [P10, P25, P50, P75, P90]
]

也可扩展为：

[
T_P = [P05, P10, P25, P50, P75, P90, P95]
]

但工程上先用 5 维即可。

### 作用

描述：

* 人物整体亮度；
* 人物阴影位置；
* 人物高光位置；
* 人物内部动态范围。

---

## 6.3 Face tone descriptor

对人脸区域提取：

[
T_F = [P10, P25, P50, P75, P90]
]

如果 face mask 不存在，跳过。

### 作用

解决“人物整体亮度接近，但人脸影调不对”的问题。

---

## 6.4 Background tone descriptor

对局部背景 ring 提取：

[
T_B = [P10, P25, P50, P75, P90]
]

---

## 6.5 人物—背景关系描述子

### 曝光关系

[
R_1 = P50_P - P50_B
]

表示人物比局部背景亮多少。

### 动态范围关系

[
R_2 = (P90-P10)_P
]

[
R_3 = (P90-P10)_B
]

必要时可增加：

[
R_4 = P75_P - P75_B
]

但先用前三项即可。

---

# 7. 距离函数设计

## 7.1 Person tone distance

[
E_P = \frac{1}{5}\sum_i |T_{P,i}^{cand} - T_{P,i}^{ref}|
]

---

## 7.2 Face tone distance

[
E_F = \frac{1}{5}\sum_i |T_{F,i}^{cand} - T_{F,i}^{ref}|
]

若没有 face：

```text
E_F = None
```

---

## 7.3 Background tone distance

[
E_B = \frac{1}{5}\sum_i |T_{B,i}^{cand} - T_{B,i}^{ref}|
]

---

## 7.4 Person-background relation distance

[
E_{PB}
======

|R_1^{cand} - R_1^{ref}|
+0.25|R_2^{cand} - R_2^{ref}|
+0.25|R_3^{cand} - R_3^{ref}|
]

权重可调，但这个初始设置比较合理：

* `R1` 最重要；
* `R2`,`R3` 次重要。

---

# 8. 最终总分

## 8.1 默认版本（推荐）

若存在可靠人脸：

[
\boxed{
E = 0.40E_P + 0.30E_{PB} + 0.20E_F + 0.10E_B
}
]

解释：

* 人物整体影调：40%
* 人物—背景关系：30%
* 人脸影调：20%
* 背景影调：10%

---

## 8.2 无人脸时

[
\boxed{
E = 0.50E_P + 0.35E_{PB} + 0.15E_B
}
]

---

## 8.3 为什么不把 color 放进主分数

当前目标是：

> 用 brightness ladder 替代生成式 pseudo-GT 的几何不稳定性。

而生成式 reference 自身可能存在：

* 天空偏蓝；
* 肤色飘；
* 背景色调波动。

如果给颜色高权重，会把 reference 的不稳定性传递到选择器。

因此建议：

* **Color = sanity check / confidence modifier**
* **Tone = main score**

---

# 9. 多人场景处理

若存在多个 person：

## 9.1 默认只选主人物

规则：

1. 面积最大；
2. face confidence 最高；
3. 距离图像中心最近；
4. 若工程上已有主人物逻辑，直接复用。

### 输出

```python
primary_person_id
primary_person_mask
```

---

## 9.2 为什么不建议一开始做多人物加权

因为需求核心是“图像中的人物”，通常关注主人物。

多人平均会引入：

* 背景路人；
* 尺寸很小的边缘人物；
* 无法稳定与 reference 对应。

因此第一版先只做主人物。

---

# 10. 可靠性机制

不能无条件选择一个 level。

## 10.1 Best level

[
k^* = \arg\min_k E_k
]

---

## 10.2 Margin

[
M = E_{2nd} - E_{best}
]

若 `M` 很小，说明多个 level 接近，选择不稳定。

---

## 10.3 Absolute threshold

设置两个阈值：

* `T_accept`
* `T_review`

规则：

```text
if E_best <= T_accept:
    status = ACCEPT
elif E_best <= T_review:
    status = REVIEW
else:
    status = REJECT
```

建议初始值：

* `T_accept = 0.18`
* `T_review = 0.30`

实际值需在少量人工样本上校准。

---

## 10.4 Confidence

定义一个简单置信度：

[
C = \exp(-E_{best}/\tau)\cdot \sigma(M)
]

其中：

* `tau` 可取 `0.15`
* `sigma(M)` 可简化为：
  [
  \min(1, M / M_0)
  ]
  例如 `M0 = 0.05`

最终输出：

```text
confidence in [0,1]
```

---

# 11. 输出内容定义

## 11.1 JSON 结果

每个 sample 输出：

```json
{
  "sample_id": "scene001",
  "best_level": "a_p050",
  "best_score": 0.083,
  "second_best_level": "a_p025",
  "second_best_score": 0.104,
  "margin": 0.021,
  "status": "ACCEPT",
  "confidence": 0.81,
  "weights": {
    "person": 0.40,
    "person_bg": 0.30,
    "face": 0.20,
    "background": 0.10
  },
  "levels": {
    "a_m100": {
      "score": 1.125,
      "person_error": 0.88,
      "person_bg_error": 1.54,
      "face_error": 0.93,
      "background_error": 0.71
    },
    "a_p050": {
      "score": 0.083,
      "person_error": 0.06,
      "person_bg_error": 0.11,
      "face_error": 0.05,
      "background_error": 0.12
    }
  }
}
```

---

## 11.2 可视化输出

### 推荐布局

```text
Row 1: Input | Reference | Best pseudo-GT
Row 2: a_m100 | a_m075 | a_m050 | a_m025
Row 3: a_000  | a_p025 | a_p050 | a_p075 | a_p100
```

### title 内容

每个 candidate title 显示：

```text
a_p050 ★ BEST
Score=0.083
P=0.06 PB=0.11 F=0.05 B=0.12
Conf=0.81
```

Reference title：

```text
Reference
(person-centered style target)
```

Input title：

```text
Source
```

Best 图 title：

```text
Aligned Pseudo-GT
a_p050
Score=0.083
```

### 可选增强

若工程方便，可额外在图上 overlay：

* person bbox / mask
* face bbox / mask
* local background ring

方便 debug。

---

# 12. 代码模块拆分建议

建议工程按以下模块实现。

---

## 12.1 `selector_config.py`

定义：

* level 顺序；
* 权重；
* 阈值；
* dilation 参数；
* face/person 选择规则；
* 可视化参数。

示例：

```python
LEVELS = [
    ("a_m100", -1.00),
    ("a_m075", -0.75),
    ("a_m050", -0.50),
    ("a_m025", -0.25),
    ("a_000",   0.00),
    ("a_p025",  0.25),
    ("a_p050",  0.50),
    ("a_p075",  0.75),
    ("a_p100",  1.00),
]

WEIGHTS_WITH_FACE = dict(person=0.40, pb=0.30, face=0.20, bg=0.10)
WEIGHTS_NO_FACE   = dict(person=0.50, pb=0.35, bg=0.15)

T_ACCEPT = 0.18
T_REVIEW = 0.30

BG_RING_INNER_RATIO = 0.02
BG_RING_OUTER_RATIO = 0.10
```

---

## 12.2 `mask_provider.py`

接口：

```python
def get_primary_person_mask(image) -> np.ndarray
def get_face_mask(image, person_mask=None) -> Optional[np.ndarray]
def get_local_background_mask(person_mask, image_shape) -> np.ndarray
```

要求：

* 对 source/ladder 使用同一套 source-aligned mask；
* 对 reference 单独做一套 mask。

---

## 12.3 `tone_descriptor.py`

接口：

```python
def rgb_to_log_luminance(img) -> np.ndarray
def compute_quantile_descriptor(logY, mask, qs=(10,25,50,75,90)) -> np.ndarray
def compute_region_dynamic_range(desc) -> float
def compute_person_bg_relation(person_desc, bg_desc) -> dict
```

---

## 12.4 `score.py`

接口：

```python
def person_tone_error(cand_desc, ref_desc) -> float
def face_tone_error(cand_desc, ref_desc) -> float
def bg_tone_error(cand_desc, ref_desc) -> float
def person_bg_relation_error(cand_rel, ref_rel) -> float
def total_score(candidate_features, reference_features, has_face=True) -> dict
```

返回结构化字典：

```python
{
    "score": ...,
    "person_error": ...,
    "person_bg_error": ...,
    "face_error": ...,
    "background_error": ...
}
```

---

## 12.5 `selector.py`

接口：

```python
def select_best_level(source, reference, ladder_dict) -> SelectionResult
```

功能：

1. 读取 / 计算 masks；
2. 为 reference 计算 descriptors；
3. 为每个 level 计算 descriptors；
4. 逐个算分；
5. 排序；
6. 决定 best / second / margin / status / confidence。

---

## 12.6 `visualize.py`

接口：

```python
def render_selection_visualization(
    source,
    reference,
    ladder_dict,
    score_dict,
    best_level,
    output_path,
    masks=None,
)
```

要求：

* 自动生成 contact sheet；
* title 显示 score 分解；
* 最佳候选用星号标记。

---

## 12.7 `run_selector.py`

命令行入口。

输入：

```bash
python run_selector.py \
  --source-root xxx/source \
  --reference-root xxx/reference \
  --ladder-root xxx/ladder \
  --output-root xxx/output
```

功能：

* 批量跑所有样本；
* 输出 pseudo_gt / visualization / scores；
* 统计 summary。

---

# 13. Summary 统计建议

整个数据集输出 summary：

```json
{
  "num_samples": 844,
  "accept_rate": 0.78,
  "review_rate": 0.16,
  "reject_rate": 0.06,
  "level_histogram": {
    "a_m100": 5,
    "a_m075": 18,
    "a_m050": 52,
    "a_m025": 81,
    "a_000": 210,
    "a_p025": 230,
    "a_p050": 160,
    "a_p075": 70,
    "a_p100": 18
  },
  "mean_best_score": 0.11,
  "mean_margin": 0.04
}
```

用于观察：

* pseudo-GT 倾向集中在哪些 level；
* 是否大量样本难以区分；
* reject 是否过多。

---

# 14. 工程实现顺序建议

## Phase 1：最小可用版本

实现：

* 主人物 mask；
* local background ring；
* person / background quantile descriptor；
* person + person/background relation score；
* best level selection；
* json + visualization。

**先不做人脸**。

### 第一版总分

[
E = 0.60E_P + 0.30E_{PB} + 0.10E_B
]

适合快速跑通。

---

## Phase 2：增强版

增加：

* face mask；
* face descriptor；
* confidence / margin；
* ACCEPT / REVIEW / REJECT；
* debug overlay。

---

## Phase 3：质量增强（可选）

增加：

* color sanity check；
* level tie-breaker；
* 多人主人物更稳定选择；
* Top-k 导出。

---

# 15. 必须避免的错误实现

1. **不要只比较全局 mean brightness**

   * 会被天空、背景主导。

2. **不要只比较 person mean**

   * 无法描述人物内部 tone structure。

3. **不要用 reference 和 candidate 做逐像素误差**

   * reference 不对齐，本质上不成立。

4. **不要把 color 作为主导指标**

   * 会把生成模型的颜色漂移传给 pseudo-GT selection。

5. **不要无条件选最小分数**

   * 需要 margin / 阈值 / confidence。

---

# 16. 最终交付给工程的核心定义

工程最终实现时，可以把任务定义成这一句：

> 对每个样本，使用 source-aligned brightness ladder 作为候选集合；以生成式 reference 为人物影调风格参考；通过人物整体影调、人脸影调、人物/局部背景关系和弱背景约束构建结构化 photometric score，选择最接近参考风格的候选 level，作为最终 aligned pseudo-GT，并输出完整可视化与评分记录。

---

如果你愿意，我下一步可以继续直接给你补两部分内容中的任意一个：

1. **工程实现伪代码 / Python 类设计骨架**
2. **给工程人员的任务说明书（接口、输入输出、验收标准）**
