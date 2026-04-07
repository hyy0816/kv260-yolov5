import cv2
import numpy as np
import vart
import xir
import sys

# ========== 配置 ==========
MODEL_PATH = "../KV260/Yolov5_kv260.xmodel"
INPUT_SIZE = 640
CONF_THRESH = 0.25
IOU_THRESH = 0.45

CLASS_NAMES = [
    'pedestrian',
    'rider',
    'car',
    'truck',
    'bus',
    'train',
    'motorcycle',
    'bicycle',
    'traffic light',
    'traffic sign',
    'other vehicle',
    'other person',
    'trailer'
]

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

def main():
    if len(sys.argv) < 2:
        print("用法：python3 inference.py <图片路径>")
        sys.exit(1)

    image_path = sys.argv[1]
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

    img_input, img_raw = preprocess(image_path)
    input_data[0] = img_input

    print("开始推理...")
    job_id = runner.execute_async(input_data, output_data)
    runner.wait(job_id)
    print("推理完成")

    results = postprocess(output_data[0], img_raw)
    print(f"检测到 {len(results)} 个目标：")
    for r in results:
        print(f"  [{r['class_name']}]  置信度：{r['score']:.3f}  位置：{r['box']}")

    result_img = draw_results(img_raw.copy(), results)
    cv2.imwrite("result.jpg", result_img)
    print("结果已保存到：result.jpg")

if __name__ == "__main__":
    main()
