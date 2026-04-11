import cv2
import numpy as np
import vart
import xir
import sys
from pathlib import Path

# ========== 配置 ==========
MODEL_PATH = "yolov5_kv260_fix.xmodel"
INPUT_SIZE = 640
CONF_THRESH = 0.6
IOU_THRESH = 0.45

CLASS_NAMES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

ANCHORS = np.array(
    [
        [[1.25000, 1.62500], [2.00000, 3.75000], [4.12500, 2.87500]],
        [[1.87500, 3.81250], [3.87500, 2.81250], [3.68750, 7.43750]],
        [[3.62500, 2.81250], [4.87500, 6.18750], [11.65625, 10.18750]],
    ],
    dtype=np.float32,
)
STRIDES = [8.0, 16.0, 32.0]

def get_child_subgraph_dpu(graph):
    root = graph.get_root_subgraph()
    children = root.toposort_child_subgraph()
    return [s for s in children if s.has_attr("device") and s.get_attr("device").upper() == "DPU"]

def preprocess(image_path):
    img_raw = cv2.imread(image_path)
    if img_raw is None:
        print(f"无法读取图片：{image_path}")
        sys.exit(1)
    img = cv2.resize(img_raw, (INPUT_SIZE, INPUT_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = (img / 255.0).astype(np.float32)
    img = np.expand_dims(img, axis=0)
    return img, img_raw

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def decode_outputs(outputs):
    dets = []
    for i, out in enumerate(outputs):
        out = out.astype(np.float32) / 4.0
        bs, ny, nx, ch = out.shape
        raw = out.reshape(ny, nx, 3, 85)
        grid_y, grid_x = np.meshgrid(np.arange(ny, dtype=np.float32), np.arange(nx, dtype=np.float32), indexing="ij")
        grid = np.stack((grid_x, grid_y), axis=-1)[:, :, None, :]

        box_xy = (sigmoid(raw[..., 0:2]) * 2.0 - 0.5 + grid) * STRIDES[i]
        box_wh = (sigmoid(raw[..., 2:4]) * 2.0) ** 2 * (ANCHORS[i][None, None, :, :] * STRIDES[i])

        obj = sigmoid(raw[..., 4:5])
        cls = sigmoid(raw[..., 5:])
        scores = obj * cls
        class_ids = scores.argmax(axis=-1)
        class_scores = scores.max(axis=-1)

        flat_boxes = np.concatenate((box_xy, box_wh), axis=-1).reshape(-1, 4)
        flat_scores = class_scores.reshape(-1)
        flat_class_ids = class_ids.reshape(-1)
        for j in range(flat_boxes.shape[0]):
            score = float(flat_scores[j])
            if score < CONF_THRESH:
                continue
            cx, cy, bw, bh = flat_boxes[j]
            class_prob = np.zeros(80, dtype=np.float32)
            class_prob[int(flat_class_ids[j])] = 1.0
            dets.append([
                float(cx),
                float(cy),
                float(bw),
                float(bh),
                score,
                *class_prob.tolist(),
            ])

    if not dets:
        return np.zeros((1, 0, 6), dtype=np.float32)
    return np.array([dets], dtype=np.float32)

def nms(boxes, scores, iou_thresh):
    return cv2.dnn.NMSBoxes(boxes, scores, CONF_THRESH, iou_thresh)

def postprocess(output, orig_img):
    h, w = orig_img.shape[:2]
    output = output[0]

    boxes, scores, class_ids = [], [], []

    for det in output:
        obj_conf = float(det[4])
        if obj_conf < CONF_THRESH:
            continue
        class_scores = det[5:]
        class_id = int(np.argmax(class_scores))
        score = obj_conf * float(class_scores[class_id])
        if score < CONF_THRESH:
            continue

        cx, cy, bw, bh = det[0], det[1], det[2], det[3]
        x1 = int((cx - bw / 2) * w / INPUT_SIZE)
        y1 = int((cy - bh / 2) * h / INPUT_SIZE)
        bw_px = int(bw * w / INPUT_SIZE)
        bh_px = int(bh * h / INPUT_SIZE)

        boxes.append([x1, y1, bw_px, bh_px])
        scores.append(score)
        class_ids.append(class_id)

    indices = nms(boxes, scores, IOU_THRESH)

    results = []
    for i in indices:
        if isinstance(i, (list, np.ndarray)):
            i = i[0]
        results.append({
            "box": boxes[i],
            "score": scores[i],
            "class_id": class_ids[i],
            "class_name": CLASS_NAMES[class_ids[i]] if class_ids[i] < len(CLASS_NAMES) else str(class_ids[i])
        })
    return results

def draw_results(img, results):
    for r in results:
        x, y, bw, bh = r["box"]
        label = f'{r["class_name"]} {r["score"]:.2f}'
        cv2.rectangle(img, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        cv2.putText(img, label, (x, max(y - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return img

def list_images(path):
    p = Path(path)
    if p.is_file():
        return [p]
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted([x for x in p.iterdir() if x.is_file() and x.suffix.lower() in exts])

def main():
    if len(sys.argv) < 2:
        print("用法：python3 inference.py <图片路径>")
        sys.exit(1)

    input_path = sys.argv[1]
    print(f"加载模型：{MODEL_PATH}")

    graph = xir.Graph.deserialize(MODEL_PATH)
    subgraphs = get_child_subgraph_dpu(graph)
    if not subgraphs:
        print("未找到 DPU 子图，请检查 xmodel 文件")
        sys.exit(1)

    runner = vart.Runner.create_runner(subgraphs[0], "run")
    input_tensors = runner.get_input_tensors()
    output_tensors = runner.get_output_tensors()

    print(f"输入张量形状：{[t.dims for t in input_tensors]}")
    print(f"输出张量形状：{[t.dims for t in output_tensors]}")

    input_data = [np.zeros(t.dims, dtype=np.float32) for t in input_tensors]
    output_data = [np.zeros(t.dims, dtype=np.float32) for t in output_tensors]

    image_list = list_images(input_path)
    if not image_list:
        print(f"未找到图片：{input_path}")
        sys.exit(1)

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    for image_path in image_list:
        img_input, img_raw = preprocess(str(image_path))
        input_data[0] = img_input

        print(f"开始推理：{image_path.name}")
        job_id = runner.execute_async(input_data, output_data)
        runner.wait(job_id)
        print("推理完成")

        decoded = decode_outputs(output_data)
        results = postprocess(decoded, img_raw)
        print(f"检测到 {len(results)} 个目标：")
        for r in results:
            print(f"  [{r['class_name']}]  置信度：{r['score']:.3f}  位置：{r['box']}")

        result_img = draw_results(img_raw.copy(), results)
        out_path = out_dir / image_path.name
        cv2.imwrite(str(out_path), result_img)
        print(f"结果已保存到：{out_path}")

if __name__ == "__main__":
    main()
