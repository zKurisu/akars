# SG2002 Akars 网球收集小车 — 优化记录

## 一、发现并解决的问题

### 1. 搜索转动卡顿（已修复）

**现象**：无目标时小车每次只转 100ms 就停止，等下一帧推理（~3秒）再转 → 卡顿

**根因**：`handle_detections()` 无检测分支里 `drive()` → `sleep(100ms)` → `standby()`，电机在推理期间完全停止

**修复**：最终方案 — 转 200ms 后停止（`SEARCH_PULSE_US=200ms`），转速从 10 降到 8 PWM（`IDLE_SPEED=8`）

### 2. 摄像头采集瓶颈（已修复）

**现象**：主循环中每帧 camera capture 耗时 ~2870ms，诊断阶段只需 ~33ms

**根因**：`get_frame()` 末尾调用 `INIT` ioctl 重置 UVC 管道，导致摄像头 AE/AWB 每帧重新收敛（~3秒）。诊断阶段连续快速捕获避开了这个问题

**修复**：将 INIT 从 `get_frame()` 末尾移到开头（`INIT → GET_FRAME` 而非 `GET_FRAME → INIT`），效果：capture **2870ms → ~60ms（47x）**，FPS **0.3 → 3~5（~15x）**

### 3. 摄像头贴到目标 / 距离震荡（已修复）

**现象**：v1 — 全速冲到球跟前，摄像头贴上去无法抓取；v2 — 在 GRAB_AREA 边界附近前后震荡

**修复**：
- 引入 `chase_speed()` 渐进减速：10% 以下全速(56)，10%~55% 线性减到 5 PWM
- `GRAB_AREA` 设为 0.55（球占满屏幕上下 = 距离合适）
- `GRAB_AREA_MAX` 设为 0.85（极端靠近才后退）
- `GRAB_CONFIRM_THRESHOLD` 从 5 降到 2
- 移除复杂的迟滞逻辑，简化为：接近 → 停 → 确认 → 抓
- `BACKWARD_PULSE_US` 从 200ms 降到 80ms（后退不再过远）

### 4. Arm 抓取流程（已重新设计）

**现象**：v1 — 车身左转 90° 后抓取（不应该转）；v2 — 机械臂向上抬起而非下探

**修复**：全新 5 步抓取流程，去掉车身转向：

| 步骤 | servo0（底座） | servo1（肩部） | servo2（夹爪） | 动作 |
|------|---------------|---------------|---------------|------|
| 1 就绪 | 150° | 140° | 120° | 臂抬起待命 |
| 2 下探 | 200° | 100° | — | 底座下摆，肩部推球 |
| 3 张开 | — | 90° | 180° | 夹爪张开，继续下压 |
| 4 夹取 | — | — | 80° | 夹爪闭合抓球 |
| 5 抬起 | 150° | 140° | — | 回到就绪位 |

舵机方向规则：servo0 减小=上抬，servo1 增大=上抬，servo2 增大=张开

### 5. Servo1/servo2 不响应（已修复）

**现象**：servo0 工作正常，servo1/servo2 在自主流程中不响应，但单独测试程序能控制

**根因（两个）**：
1. **指令缺换行符**：ZP10S 协议要求 `\r\n` 结尾。之前 `#000PxxxxT1000!` 无换行，连续多条粘在一起 → 控制器解析失败。测试程序只发一条指令不触发此问题
2. **指令间隔过短**：连续 `set_angle` 之间只有 50ms 延迟，控制器来不及处理

**修复**：
- 所有指令末尾加 `\r\n`（`set_angle`、`restore_torque`、`release_torque`）
- 连续指令间隔从 50ms 增大到 300ms

### 6. 预处理性能（已部分优化）

**现象**：YUV→RGB 预处理耗时 ~59ms，占总帧时 18%

**根因**：`yuv422p_to_rgb_planar()` 每次先清零整个 640×640×3≈1.2MB 输出 buffer，而实际只有 letterbox 上下黑边需要清零

**修复**：只清零 padding 区域（top/bottom 各 80 行 × 3 平面 = ~300KB），省 ~4ms（59→55ms）。进一步优化空间：降低模型输入分辨率（需重新训练模型）

---

## 二、新增功能

### 摄像头诊断（`camera.rs`）

新增 `diagnose_timing()` 方法，启动时自动运行：分别用 cmd3（原始 MJPEG）和 cmd4（JPU 解码 YUV）各捕获一帧，计算帧等待时间 + JPU 解码开销对比

### Web 控制台摄像头画面（`web.rs` + `main.rs`）

- 新增 `/camera.jpg` 端点，返回最新帧 JPEG（200ms 轮询刷新）
- `serve` 命令新增 `--camera` 选项启动后台采集线程
- HTML 控制台顶部新增摄像头实时画面区域

### 舵机角度标定工具（8 个 bin 程序）

`src/bin/test_servo{0,1,2}_{lift,prepare,approach,grab}.rs`，每个支持 `--angle` 命令行参数，默认值对应当前配置角度

### 调试信息增强（`robot.rs`）

- 启动时打印 `[cfg]` 行：所有关键常量值
- `[time]` 行新增 `handle`（检测处理耗时）和 `total`（总帧耗时）
- `[FPS]` 行新增 `status`、`area`、`cx`、`confirm` 字段

---

## 三、文件改动清单

| 文件 | 主要改动 |
|------|---------|
| `src/camera.rs` | INIT 从 post-frame 移到 pre-frame；新增 `diagnose_timing()` |
| `src/image_bridge.rs` | Letterbox 清零优化；新增 `yuv422p_to_jpeg_bytes()` |
| `src/robot.rs` | `chase_speed()` 渐进减速；简化决策树；调整 GRAB_AREA 等常量；增强 `[time]`/`[FPS]`/`[cfg]` 输出 |
| `src/arm.rs` | 全新 5 步抓取流程；指令加 `\r\n`；增大指令间隔；更新所有角度常量 |
| `src/web.rs` | AppState 新增 camera_frame；新增 `/camera.jpg` 端点；HTML 新增摄像头区域 |
| `src/main.rs` | `serve` 命令新增 `--camera` 选项 |
| `Cargo.toml` | 新增 8 个 test_servo* bin 目标 |
| `src/bin/test_servo*.rs` | 8 个舵机角度测试程序（新增） |

---

## 四、当前关键常量

```
GRAB_AREA       = 0.55    # 球占画面 55% 时触发抓取
GRAB_AREA_MAX   = 0.85    # 极端靠近后退阈值
CHASE_SPEED     = 56      # 追击全速
CONFIRM         = 2       # 抓取确认帧数
IDLE            = 8       # 搜索转速
SEARCH_PULSE    = 200ms   # 搜索转动脉冲
BACKWARD_PULSE  = 80ms    # 后退脉冲
CENTER_MARGIN   = 35px    # 居中判定容差
```

## 五、当前帧耗时分布

| 阶段 | 耗时 | 占比 |
|------|------|------|
| cap 采集 | ~60ms | 20% |
| pre 预处理 | ~55ms | 18% |
| fwd TPU推理 | ~43ms | 14% |
| post 后处理 | ~14ms | 5% |
| handle 控制 | 10~150ms | 3-50% |
| **总计** | **190~330ms** | **3-5 FPS** |
