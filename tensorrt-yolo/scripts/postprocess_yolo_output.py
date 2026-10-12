import argparse
import numpy as np


OUTPUT_SHAPE = (1, 84, 8400)

PERSON_CLASS_ID = 0

CONF_THRESHOLD = 0.7
IOU_THRESHOLD = 0.7

# 当前固定 ONNX 输入
MODEL_WIDTH = 640
MODEL_HEIGHT = 640

# 当前真实业务图尺寸
BUSINESS_WIDTH = 360
BUSINESS_HEIGHT = 640

# 360x640 → 640x640
# ratio = 1
# 左右各 pad 140
LETTERBOX_RATIO = 1.0
PAD_X = 140.0
PAD_Y = 0.0


def load_output(path):
    output = np.fromfile(
        path,
        dtype="<f4",
    )

    expected = np.prod(OUTPUT_SHAPE)

    if output.size != expected:
        raise RuntimeError(
            f"Invalid output size: "
            f"{output.size}, expected {expected}"
        )

    output = output.reshape(OUTPUT_SHAPE)

    if not np.isfinite(output).all():
        raise RuntimeError(
            "Output contains NaN or Inf"
        )

    return output


def xywh_to_xyxy(boxes):
    """
    boxes:
        [N, 4]
        x_center, y_center, width, height

    return:
        [N, 4]
        x1, y1, x2, y2
    """

    result = np.empty_like(boxes)

    result[:, 0] = (
        boxes[:, 0] - boxes[:, 2] / 2.0
    )

    result[:, 1] = (
        boxes[:, 1] - boxes[:, 3] / 2.0
    )

    result[:, 2] = (
        boxes[:, 0] + boxes[:, 2] / 2.0
    )

    result[:, 3] = (
        boxes[:, 1] + boxes[:, 3] / 2.0
    )

    return result


def box_iou_one_to_many(box, boxes):
    """
    box:
        [4]

    boxes:
        [N, 4]

    return:
        IoU [N]
    """

    if boxes.shape[0] == 0:
        return np.empty(
            (0,),
            dtype=np.float32,
        )

    x1 = np.maximum(
        box[0],
        boxes[:, 0],
    )

    y1 = np.maximum(
        box[1],
        boxes[:, 1],
    )

    x2 = np.minimum(
        box[2],
        boxes[:, 2],
    )

    y2 = np.minimum(
        box[3],
        boxes[:, 3],
    )

    inter_w = np.maximum(
        0.0,
        x2 - x1,
    )

    inter_h = np.maximum(
        0.0,
        y2 - y1,
    )

    intersection = (
        inter_w * inter_h
    )

    area_a = (
        (box[2] - box[0]) *
        (box[3] - box[1])
    )

    area_b = (
        (boxes[:, 2] - boxes[:, 0]) *
        (boxes[:, 3] - boxes[:, 1])
    )

    union = (
        area_a +
        area_b -
        intersection
    )

    return intersection / np.maximum(
        union,
        1e-12,
    )


def nms(
    boxes,
    scores,
    iou_threshold,
):
    """
    boxes:
        [N, 4]

    scores:
        [N]

    return:
        被保留下来的索引
    """

    order = np.argsort(
        scores
    )[::-1]

    keep = []

    while order.size > 0:

        current = order[0]

        keep.append(current)

        if order.size == 1:
            break

        rest = order[1:]

        ious = box_iou_one_to_many(
            boxes[current],
            boxes[rest],
        )

        rest = rest[
            ious <= iou_threshold
        ]

        order = rest

    return np.asarray(
        keep,
        dtype=np.int64,
    )


def restore_letterbox(boxes):
    """
    640x640 模型坐标
        ↓
    360x640 业务图坐标
    """

    boxes = boxes.copy()

    # 去掉 padding
    boxes[:, [0, 2]] -= PAD_X
    boxes[:, [1, 3]] -= PAD_Y

    # 除以 resize ratio
    boxes[:, [0, 2]] /= LETTERBOX_RATIO
    boxes[:, [1, 3]] /= LETTERBOX_RATIO

    # clip 到业务图范围
    boxes[:, [0, 2]] = np.clip(
        boxes[:, [0, 2]],
        0.0,
        BUSINESS_WIDTH,
    )

    boxes[:, [1, 3]] = np.clip(
        boxes[:, [1, 3]],
        0.0,
        BUSINESS_HEIGHT,
    )

    return boxes


def postprocess(
    output,
    conf_threshold=CONF_THRESHOLD,
    iou_threshold=IOU_THRESHOLD,
):
    """
    YOLOv8 raw output:
        [1, 84, 8400]

    return:
        final detections
    """

    # ---------------------------------
    # 1. 取 batch 0
    # ---------------------------------

    output = output[0]

    # [4, 8400]
    bbox = output[:4, :]

    # [80, 8400]
    cls = output[4:, :]

    # ---------------------------------
    # 2. 每个 candidate 选最高类别
    # ---------------------------------

    class_ids = np.argmax(
        cls,
        axis=0,
    )

    scores = np.max(
        cls,
        axis=0,
    )

    # ---------------------------------
    # 3. 历史业务过滤
    #
    # 先：
    #   argmax class
    #
    # 再：
    #   class == person
    #   score >= 0.7
    # ---------------------------------

    mask = (
        (class_ids == PERSON_CLASS_ID) &
        (scores >= conf_threshold)
    )

    candidate_indices = np.where(
        mask
    )[0]

    if candidate_indices.size == 0:
        return []

    # ---------------------------------
    # 4. 取对应 bbox
    #
    # 原来:
    # [4, 8400]
    #
    # 变成:
    # [N, 4]
    # ---------------------------------

    boxes = bbox[
        :,
        candidate_indices
    ].T

    selected_scores = scores[
        candidate_indices
    ]

    # ---------------------------------
    # 5. xywh → xyxy
    # ---------------------------------

    boxes = xywh_to_xyxy(
        boxes
    )

    # ---------------------------------
    # 6. NMS
    # ---------------------------------

    keep = nms(
        boxes,
        selected_scores,
        iou_threshold,
    )

    boxes = boxes[keep]

    selected_scores = (
        selected_scores[keep]
    )

    candidate_indices = (
        candidate_indices[keep]
    )

    # ---------------------------------
    # 7. LetterBox 坐标还原
    # ---------------------------------

    restored_boxes = restore_letterbox(
        boxes
    )

    # ---------------------------------
    # 8. 生成结果
    # ---------------------------------

    detections = []

    for i in range(
        len(selected_scores)
    ):
        detections.append(
            {
                "candidate": int(
                    candidate_indices[i]
                ),
                "class_id": 0,
                "class_name": "person",
                "confidence": float(
                    selected_scores[i]
                ),
                "box_model": boxes[
                    i
                ].copy(),
                "box_business": restored_boxes[
                    i
                ].copy(),
            }
        )

    return detections


def print_detections(
    name,
    detections,
):
    print(
        f"\n=== {name} ==="
    )

    print(
        "Detections:",
        len(detections),
    )

    for i, det in enumerate(
        detections
    ):
        print(
            f"\nDetection {i}"
        )

        print(
            "candidate :",
            det["candidate"],
        )

        print(
            "class     :",
            det["class_name"],
        )

        print(
            "confidence:",
            f"{det['confidence']:.8f}",
        )

        print(
            "model box :",
            np.round(
                det["box_model"],
                4,
            ),
        )

        print(
            "business  :",
            np.round(
                det["box_business"],
                4,
            ),
        )


def box_iou_pair(
    a,
    b,
):
    a = np.asarray(
        a,
        dtype=np.float32,
    )

    b = np.asarray(
        b,
        dtype=np.float32,
    )

    x1 = max(
        a[0],
        b[0],
    )

    y1 = max(
        a[1],
        b[1],
    )

    x2 = min(
        a[2],
        b[2],
    )

    y2 = min(
        a[3],
        b[3],
    )

    inter_w = max(
        0.0,
        x2 - x1,
    )

    inter_h = max(
        0.0,
        y2 - y1,
    )

    intersection = (
        inter_w * inter_h
    )

    area_a = (
        (a[2] - a[0]) *
        (a[3] - a[1])
    )

    area_b = (
        (b[2] - b[0]) *
        (b[3] - b[1])
    )

    union = (
        area_a +
        area_b -
        intersection
    )

    return intersection / max(
        union,
        1e-12,
    )


def compare_detections(
    fp32_detections,
    fp16_detections,
):
    print(
        "\n=== FP32 vs FP16 Final Detections ==="
    )

    print(
        "FP32 count:",
        len(fp32_detections),
    )

    print(
        "FP16 count:",
        len(fp16_detections),
    )

    if (
        len(fp32_detections) !=
        len(fp16_detections)
    ):
        print(
            "Detection count match: NO"
        )

        return

    print(
        "Detection count match: YES"
    )

    for i, (
        det32,
        det16,
    ) in enumerate(
        zip(
            fp32_detections,
            fp16_detections,
        )
    ):
        iou = box_iou_pair(
            det32["box_business"],
            det16["box_business"],
        )

        conf_diff = abs(
            det32["confidence"] -
            det16["confidence"]
        )

        print(
            f"\nDetection {i}"
        )

        print(
            "candidate FP32:",
            det32["candidate"],
        )

        print(
            "candidate FP16:",
            det16["candidate"],
        )

        print(
            "confidence diff:",
            conf_diff,
        )

        print(
            "box IoU:",
            iou,
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fp32",
        default=(
            "data/outputs/"
            "tensorrt_fp32_output.bin"
        ),
    )

    parser.add_argument(
        "--fp16",
        default=(
            "data/outputs/"
            "tensorrt_fp16_output.bin"
        ),
    )

    args = parser.parse_args()

    fp32_output = load_output(
        args.fp32
    )

    fp16_output = load_output(
        args.fp16
    )

    fp32_detections = postprocess(
        fp32_output
    )

    fp16_detections = postprocess(
        fp16_output
    )

    print_detections(
        "TensorRT FP32",
        fp32_detections,
    )

    print_detections(
        "TensorRT FP16",
        fp16_detections,
    )

    compare_detections(
        fp32_detections,
        fp16_detections,
    )


if __name__ == "__main__":
    main()