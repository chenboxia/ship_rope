
## 硬件要求

| 项目 | 要求 |
|------|------|
| CPU | >=8核, >=2.2GHz, 国产自主可控 |
| 内存 | >=32GB DDR4 |
| GPU | 显存>=8GB, >=100 TOPS INT8 |
| 温度 | -10C ~ 60C |
| 可靠性 | >=1万小时无故障 |

## 项目结构

```
系缆/
├── README.md
├── requirements.txt
├── configs/config.yaml
├── data/dataset.yaml              # 船/缆绳/船员数据集
├── data/lifejacket_dataset.yaml   # 救生衣数据集
├── src/
│   ├── geometry.py          # 几何计算
│   ├── detector.py          # YOLO11s检测器
│   ├── rope_analyzer.py     # 缆绳形态分析（三级端点提取+三维判定）
│   ├── behavior_verifier.py # 船员行为验证
│   ├── state_machine.py     # 五阶段状态机
│   ├── lifejacket_detector.py # 救生衣识别
│   ├── data_exporter.py     # 结构化JSON输出
│   ├── monitor.py           # 主监测流水线
│   ├── engine.py            # 工业引擎（多路流/watchdog/GPU监控/热更新）
│   ├── stream_manager.py    # 多路摄像头管理（断流重连/心跳）
│   ├── model_optimizer.py   # TensorRT/ONNX导出+INT8量化
│   ├── api_server.py        # HTTP API（对接综合辅助终端）
│   ├── trainer.py           # 训练模块
│   ├── inference.py         # 推理引擎
│   └── visualizer.py        # 可视化
├── scripts/
│   ├── train.py             # 系缆模型训练
│   ├── train_lifejacket.py  # 救生衣模型训练
│   ├── run_inference.py     # 推理入口（支持多路）
│   ├── export_model.py      # 模型导出（ONNX/TensorRT/基准测试）
│   └── calibrate.py         # 壁面区域标定
├── weights/
└── outputs/
```

## 使用流程

### 1. 训练模型

```bash
# 系缆检测模型（标注 ship/rope/crew 三类）
python scripts/train.py --epochs 100 --batch 16

# 救生衣模型（标注 lifejacket/no_lifejacket 两类）
python scripts/train_lifejacket.py --epochs 80 --batch 16
```

### 2. 导出加速模型

```bash
# ONNX导出
python scripts/export_model.py --weights weights/best.pt --format onnx

# TensorRT INT8导出（适配100 TOPS硬件）
python scripts/export_model.py --weights weights/best.pt --format engine --int8

# 基准测试
python scripts/export_model.py --weights weights/best.pt --benchmark
```

### 3. 标定壁面区域

```bash
python scripts/calibrate.py --source video.mp4
```

### 4. 部署运行

```bash
# 单路摄像头
python scripts/run_inference.py --source 0 --show

# 多路摄像头（前后锚桩）
python scripts/run_inference.py --source cam_front.mp4 --source cam_rear.mp4

# RTSP流
python scripts/run_inference.py --source rtsp://192.168.1.10/stream
```

## 工业部署能力

| 能力 | 说明 |
|------|------|
| 多路并行 | 支持多路摄像头同时采集处理 |
| 断流重连 | 摄像头断开自动重连，指数退避 |
| 心跳检测 | 帧超时自动判定断流 |
| GPU监控 | 显存使用率监控，OOM预警+自动清理 |
| 配置热更新 | 修改config.yaml自动生效，无需重启 |
| 进程守护 | SIGINT/SIGTERM优雅停机 |
| 工业日志 | loguru日志轮转，异常自动记录 |
| HTTP API | GET /api/latest /api/status /health |
| 模型加速 | 支持TensorRT/ONNX INT8量化 |
| 结构化输出 | JSON数据对接船舶过闸综合辅助终端 |

## API接口

推理运行时默认开启HTTP服务（端口8080）：

```
GET /health          - 健康检查
GET /api/status      - 系统状态
GET /api/latest      - 最新一帧监测结果
GET /api/history?n=10 - 最近N帧历史
```

## 状态说明

- **MOORED**（绿色）：缆绳张紧且与船体连接正常
- **UNMOORED**（红色）：脱系超时无船员操作，触发报警
- **SWITCHING**（黄色）：缆绳脱系但检测到船员正在操作
- **MONITORING**（黄色）：脱系中，等待确认
