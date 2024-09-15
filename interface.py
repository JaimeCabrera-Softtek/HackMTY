import streamlit as st
import collections
import time
import cv2
import numpy as np
import openvino as ov
from ultralytics.yolo.utils import ops
from ultralytics.yolo.utils.plotting import colors
import torch
from ultralytics import YOLO
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Tuple, Dict

# Load YOLO model and OpenVINO core
@st.cache_resource
def load_models():
    det_model = YOLO('yolov8n.pt')
    core = ov.Core()
    quantized_det_model = core.read_model("models/yolov8n_openvino_model/yolov8n.xml")
    return det_model, core, quantized_det_model

det_model, core, quantized_det_model = load_models()
label_map = det_model.model.names

# Helper functions
def plot_one_box(box:np.ndarray, img:np.ndarray,
                 color:Tuple[int, int, int] = None,
                 label:str = None, line_thickness:int = 5):
    """
    Helper function for drawing single bounding box on image
    """
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return img

def draw_results(results:Dict, source_image:np.ndarray, label_map:Dict):
    """
    Helper function for drawing bounding boxes on image
    """
    boxes = results["det"]
    for idx, (*xyxy, conf, lbl) in enumerate(boxes):
        if int(lbl) == 0:
            label = f'{label_map[int(lbl)]} {conf:.2f}'
            source_image = plot_one_box(xyxy, source_image, label=label, color=colors(int(lbl)), line_thickness=1)
    return source_image

def postprocess(
    pred_boxes:np.ndarray,
    input_hw:Tuple[int, int],
    orig_img:np.ndarray,
    min_conf_threshold:float = 0.25,
    nms_iou_threshold:float = 0.7,
    agnosting_nms:bool = False,
    max_detections:int = 300,
):
    """
    YOLOv8 model postprocessing function
    """
    nms_kwargs = {"agnostic": agnosting_nms, "max_det":max_detections}
    preds = ops.non_max_suppression(
        torch.from_numpy(pred_boxes),
        min_conf_threshold,
        nms_iou_threshold,
        nc=80,
        **nms_kwargs
    )

    results = []
    for i, pred in enumerate(preds):
        shape = orig_img[i].shape if isinstance(orig_img, list) else orig_img.shape
        if not len(pred):
            results.append({"det": [], "segment": []})
            continue
        pred[:, :4] = ops.scale_boxes(input_hw, pred[:, :4], shape).round()
        results.append({"det": pred})
    return results

def letterbox(img: np.ndarray, new_shape: Tuple[int, int] = (640, 640), color: Tuple[int, int, int] = (114, 114, 114),
              auto: bool = False, scale_fill: bool = False, scaleup: bool = False, stride: int = 32):
    """
    Resize image and padding for detection
    """
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scale_fill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

def preprocess_image(img0: np.ndarray):
    """
    Preprocess image according to YOLOv8 input requirements
    """
    img = letterbox(img0)[0]
    img = img.transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    return img

def image_to_tensor(image: np.ndarray):
    """
    Convert image to tensor format
    """
    input_tensor = image.astype(np.float32)  # uint8 to fp32
    input_tensor /= 255.0  # 0 - 255 to 0.0 - 1.0
    if input_tensor.ndim == 3:
        input_tensor = np.expand_dims(input_tensor, 0)
    return input_tensor

def detect(image:np.ndarray, model:ov.Model):
    """
    OpenVINO YOLOv8 model inference function
    """
    preprocessed_image = preprocess_image(image)
    input_tensor = image_to_tensor(preprocessed_image)
    result = model(input_tensor)
    boxes = result[model.output(0)]
    input_hw = input_tensor.shape[2:]
    detections = postprocess(pred_boxes=boxes, input_hw=input_hw, orig_img=image)
    return detections

def run_object_detection(video_source, model, device="AUTO"):
    ov_config = {}
    if "GPU" in device or ("AUTO" in device and "GPU" in core.available_devices):
        ov_config = {"GPU_DISABLE_WINOGRAD_CONVOLUTION": "YES"}
    quantized_det_compiled_model = core.compile_model(model, "AUTO", ov_config)

    cap = cv2.VideoCapture(video_source)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Streamlit placeholders
    video_placeholder = st.empty()
    heatmap_placeholder = st.empty()

    df = pd.DataFrame(columns=['x', 'y'])
    processing_times = collections.deque(maxlen=200)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Resize frame if necessary
        scale = 1280 / max(frame.shape)
        if scale < 1:
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        input_image = np.array(frame)

        start_time = time.time()
        detections = detect(input_image[:, :, ::-1], quantized_det_compiled_model)[0]
        stop_time = time.time()

        for detection in detections['det']:
            if detection[5] == 0:  # Class 0 is person
                x_center = (detection[0] + detection[2]) / 2
                y_center = (detection[1] + detection[3]) / 2
                df = pd.concat([df, pd.DataFrame({'x': [x_center], 'y': [y_center]})], ignore_index=True)

        image_with_boxes = draw_results(detections, input_image, {0: 'person'})

        processing_time = (stop_time - start_time) * 1000
        fps = 1000 / processing_time
        processing_times.append(processing_time)

        cv2.putText(image_with_boxes, f"Inference time: {processing_time:.1f}ms ({fps:.1f} FPS)",
                    (20, 40), cv2.FONT_HERSHEY_COMPLEX, frame.shape[1] / 1000, (0, 0, 255), 1, cv2.LINE_AA)

        # Update video frame
        video_placeholder.image(image_with_boxes, channels="BGR", use_column_width=True)

        # Update heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.kdeplot(data=df, x="x", y="y", cmap="YlOrRd", fill=True, cbar=True, ax=ax)
        ax.set_xlim(0, frame_width)
        ax.set_ylim(frame_height, 0)
        ax.set_title("Real-time Heatmap of Person Detections")
        heatmap_placeholder.pyplot(fig)
        plt.close(fig)

    cap.release()

def main():
    st.title("YOLOv8 OpenVINO Person Detection with Real-time Heatmap")

    # Choose between webcam and video upload
    source_option = st.radio("Select input source:", ("Webcam", "Upload Video"))

    if source_option == "Webcam":
        if st.button("Start Webcam"):
            run_object_detection(0, quantized_det_model)  # 0 is typically the default webcam
    else:
        # File uploader for video
        video_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
        if video_file is not None:
            # Save uploaded file temporarily
            with open("temp_video.mp4", "wb") as f:
                f.write(video_file.read())
            # Process the video
            run_object_detection("temp_video.mp4", quantized_det_model)
        else:
            st.write("Please upload a video file.")

if __name__ == "__main__":
    main()