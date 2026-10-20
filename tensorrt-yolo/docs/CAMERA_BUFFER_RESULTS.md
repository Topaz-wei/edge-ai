# USB 摄像头缓冲修改与验证

当前默认采集缓冲由1改为2，并新增 `--capture-buffers N`。设置失败或实际缓冲数不符会报错，日志记录 requested/actual。

## 完整推理比较

1280×720 MJPG/30FPS，同一个 mixed engine，Graph开启；不显示不保存。2/3/4、4/3/2顺序各测两轮，每轮180帧、排除前30帧。共900张稳态计时样本。

| 缓冲 | 实际FPS/QPS | 读到时距驱动时间戳均值ms | 处理完成时距驱动时间戳均值ms | 完成间隔P95 ms |
|---|---:|---:|---:|---:|
| 2 | 29.989 | 41.233 | 55.737 | 61.415 |
| 3 | 30.003 | 41.351 | 55.940 | 60.996 |
| 4 | 30.009 | 41.442 | 55.587 | 60.641 |

三种缓冲都达到约30FPS，更大的缓冲没有增加吞吐，因此选择其中最小的2作为本设备默认值。各轮有效窗口前30帧和后30帧的平均read_age差值均小于0.6ms，短时测试未发现持续增加的积压趋势。

这些间隔由CLOCK_MONOTONIC减去OpenCV/V4L2帧时间戳得到。程序检查非零、有限、非未来且差值不超过10秒，并用CSV确认时间戳单调增加、帧间约33.3ms。它不是精确的曝光到显示器延迟；时间戳来源、曝光时刻、屏幕刷新与面板延迟未用外部设备测量。

## 默认值回归

额外使用默认参数（不传capture-buffers）验证真实桌面显示和MP4保存，每种180帧、去掉前30帧：

- display: 30.030 FPS/QPS；日志确认 requested=2, actual=2。
- save: 28.900 FPS/QPS；日志确认 requested=2, actual=2。

输出MP4重新解码确认180帧、360×640。编译和四项CTest通过。没有改动engine、分辨率、转置缩放、遮罩、阈值、采集格式和Graph模式。

## 使用

原命令无需增加参数即可使用默认2缓冲；也可显式指定：

```bash
build/yolo_video_cuda \
  debug/trial007_sub_model22/yolov8m_int8.plan \
  /dev/v4l/by-id/usb-LRCP_F1080P_LRCP_F1080P_000000000001-video-index0 camera \
  --capture-width 1280 --capture-height 720 \
  --capture-fps 30 --capture-fourcc MJPG \
  --capture-buffers 2
```

持续处理如果慢于供帧速度，仍可能有旧帧；本次没有加入丢帧线程，未声称所有负载和所有摄像头下都能保证30FPS或低延迟。首次engine初始化可能使采集队列堆积，以上稳态结论不包含启动阶段。

## 本地证据

`debug/camera-buffers-v1/`：run.py、experiment.json、6份逐帧CSV及日志、summary.json、aggregate.json、default-display/default-save CSV和日志、default_mode_checks.json。摄像头视频留在本地，不进入Git。旧单缓冲报告保留，供比较，不用新结果覆盖。
