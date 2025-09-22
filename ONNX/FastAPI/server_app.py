# server_app.py
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
import cv2
import numpy as np
import onnxruntime as ort
import time
from collections import deque
import threading

app = FastAPI(title="YOLOv8 ONNX Server")

# 성능 메트릭 추적을 위한 전역 변수
class PerformanceTracker:
    def __init__(self):
        self.frame_times = deque(maxlen=30)  # 최근 30프레임의 시간 저장
        self.bandwidth_data = deque(maxlen=30)  # 최근 30프레임의 대역폭 데이터
        self.last_frame_time = time.time()
        self.frame_count = 0
        self.lock = threading.Lock()
        
    def update_frame(self, frame_size, process_time):
        current_time = time.time()
        with self.lock:
            # FPS 계산을 위한 프레임 시간 저장
            if self.frame_count > 0:
                frame_interval = current_time - self.last_frame_time
                self.frame_times.append(frame_interval)
            
            # 대역폭 계산 (bytes per second)
            if len(self.frame_times) > 0:
                avg_frame_time = sum(self.frame_times) / len(self.frame_times)
                bandwidth = frame_size / avg_frame_time if avg_frame_time > 0 else 0
                self.bandwidth_data.append(bandwidth)
            
            self.last_frame_time = current_time
            self.frame_count += 1
            
    def get_fps(self):
        with self.lock:
            if len(self.frame_times) == 0:
                return 0.0
            avg_frame_time = sum(self.frame_times) / len(self.frame_times)
            return 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
    
    def get_bandwidth(self):
        with self.lock:
            if len(self.bandwidth_data) == 0:
                return 0.0
            return sum(self.bandwidth_data) / len(self.bandwidth_data)

# 성능 추적기 인스턴스
perf_tracker = PerformanceTracker()

# 1. ONNX 모델 및 클래스 이름 로드
try:
    # ONNX 모델의 경로를 프로젝트 구조에 맞게 수정하세요.
    yolo_model_path = "models/yolov8n.onnx"
    # CPU 실행을 기본으로 설정합니다. GPU 사용 시 providers=['CUDAExecutionProvider']
    ort_session = ort.InferenceSession(yolo_model_path, providers=['CPUExecutionProvider'])
    input_size = ort_session.get_inputs()[0].shape[2]
    print(f"ONNX 모델 로드 성공! 입력 크기: {input_size}x{input_size}")
    
    # COCO 데이터셋 클래스 이름 (80개)
    class_names = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "bottle", "wine glass",
        "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
        "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
        "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster",
        "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    ]

except Exception as e:
    print(f"ONNX 모델 로드 중 오류 발생: {e}")
    ort_session = None
    input_size = 640
    class_names = [f"class_{i}" for i in range(80)]


def preprocess(image: np.ndarray, size: int):
    """YOLOv8 ONNX 입력 크기에 맞게 이미지 전처리"""
    h, w = image.shape[:2]
    scale = size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    padded_img = np.full((size, size, 3), 114, dtype=np.uint8)
    padded_img[:new_h, :new_w] = resized_img

    padded_img = padded_img.astype(np.float32) / 255.0
    padded_img = padded_img.transpose(2, 0, 1)
    input_tensor = np.expand_dims(padded_img, axis=0)
    return input_tensor, scale, new_w, new_h


def postprocess(predictions, scale, new_w, new_h, orig_w, orig_h,
                conf_thres=0.5, iou_thres=0.45):
    """YOLOv8 ONNX 출력 후처리"""
    # YOLOv8 ONNX 출력: (1, 84, N) -> (N, 84)로 변환
    output = np.squeeze(predictions[0]).T

    # 박스 좌표와 클래스 점수 분리
    boxes = output[:, :4] 
    class_scores = output[:, 4:]
    
    # 최대 클래스 점수와 클래스 ID 계산
    max_scores = np.max(class_scores, axis=1)
    class_ids = np.argmax(class_scores, axis=1)
    
    # 신뢰도 필터링
    valid_mask = max_scores >= conf_thres
    
    # 유효한 감지가 없으면 빈 리스트 반환
    if not np.any(valid_mask):
        return []

    filtered_boxes = boxes[valid_mask]
    filtered_scores = max_scores[valid_mask]
    filtered_class_ids = class_ids[valid_mask]

    # (cx, cy, w, h) → (x1, y1, x2, y2) 변환
    boxes_xyxy = np.zeros_like(filtered_boxes)
    boxes_xyxy[:, 0] = filtered_boxes[:, 0] - filtered_boxes[:, 2] / 2
    boxes_xyxy[:, 1] = filtered_boxes[:, 1] - filtered_boxes[:, 3] / 2
    boxes_xyxy[:, 2] = filtered_boxes[:, 0] + filtered_boxes[:, 2] / 2
    boxes_xyxy[:, 3] = filtered_boxes[:, 1] + filtered_boxes[:, 3] / 2
    
    # NMS 적용 (NMSBoxes는 x1, y1, x2, y2 형태의 리스트를 요구)
    indices = cv2.dnn.NMSBoxes(
        boxes_xyxy.tolist(),
        filtered_scores.tolist(),
        conf_thres,
        iou_thres
    )

    detections = []
    if len(indices) > 0:
        for i in indices.flatten():
            box = boxes_xyxy[i]
            score = filtered_scores[i]
            class_id = filtered_class_ids[i]
            
            # 패딩 역변환을 고려하여 원본 이미지 크기로 좌표를 변환
            pad_x = (input_size - new_w) / 2
            pad_y = (input_size - new_h) / 2
            
            x1 = int(np.clip((box[0] - pad_x) / scale, 0, orig_w))
            y1 = int(np.clip((box[1] - pad_y) / scale, 0, orig_h))
            x2 = int(np.clip((box[2] - pad_x) / scale, 0, orig_w))
            y2 = int(np.clip((box[3] - pad_y) / scale, 0, orig_h))
            
            if x2 > x1 and y2 > y1:
                detections.append({
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'confidence': float(score), 
                    'class_id': int(class_id), 
                    'name': class_names[class_id]
                })

    return detections


def format_bytes(bytes_value):
    """바이트를 읽기 쉬운 형태로 변환"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f} TB"


def format_bandwidth(bandwidth):
    """대역폭을 읽기 쉬운 형태로 변환 (bits per second)"""
    bandwidth_bps = bandwidth * 8  # bytes to bits
    for unit in ['bps', 'Kbps', 'Mbps', 'Gbps']:
        if bandwidth_bps < 1000.0:
            return f"{bandwidth_bps:.1f} {unit}"
        bandwidth_bps /= 1000.0
    return f"{bandwidth_bps:.1f} Tbps"


def draw_performance_info(frame, fps, bandwidth, frame_size, process_time):
    """프레임 왼쪽 상단에 성능 정보를 그리기"""
    # 배경 박스 그리기
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (350, 130), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # 텍스트 정보
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    color = (0, 255, 255)  # 노란색
    thickness = 1
    
    # FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (15, 35), font, font_scale, color, thickness)
    
    # Bandwidth
    bandwidth_str = format_bandwidth(bandwidth)
    cv2.putText(frame, f"Bandwidth: {bandwidth_str}", (15, 60), font, font_scale, color, thickness)
    
    # Frame Size
    size_str = format_bytes(frame_size)
    cv2.putText(frame, f"Frame Size: {size_str}", (15, 85), font, font_scale, color, thickness)
    
    # Processing Time
    cv2.putText(frame, f"Process Time: {process_time:.1f}ms", (15, 110), font, font_scale, color, thickness)


@app.post("/video")
async def video_endpoint(request: Request):
    """클라이언트에서 전송한 이미지 프레임을 받아 추론"""
    if ort_session is None:
        return JSONResponse({"error": "ONNX model not loaded."}, status_code=500)

    try:
        start_time = time.time()
        contents = await request.body()
        frame_size = len(contents)  # 전송된 프레임 크기
        
        np_array = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

        if frame is None:
            return JSONResponse({"status": "received", "detections": []})

        orig_h, orig_w = frame.shape[:2]

        # 추론 시작 시간
        inference_start = time.time()
        
        input_tensor, scale, new_w, new_h = preprocess(frame, input_size)
        input_name = ort_session.get_inputs()[0].name
        predictions = ort_session.run(None, {input_name: input_tensor})
        
        detections = postprocess(predictions, scale, new_w, new_h, orig_w, orig_h)
        
        # 처리 시간 계산
        process_time = (time.time() - inference_start) * 1000  # 밀리초로 변환
        
        # 성능 메트릭 업데이트
        perf_tracker.update_frame(frame_size, process_time / 1000)
        
        # 현재 성능 메트릭 가져오기
        current_fps = perf_tracker.get_fps()
        current_bandwidth = perf_tracker.get_bandwidth()
        
        # 결과 표시 (디버깅용)
        display_frame = frame.copy()
        
        # 객체 감지 결과 그리기
        for det in detections:
            x1, y1, x2, y2 = det['x1'], det['y1'], det['x2'], det['y2']
            label = f"{det['name']}: {det['confidence']:.2f}"
            color = (0, 255, 0)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(display_frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 성능 정보 그리기
        draw_performance_info(display_frame, current_fps, current_bandwidth, 
                            frame_size, process_time)

        try:
            cv2.imshow("YOLOv8 Inference", display_frame)
            cv2.waitKey(1)
        except cv2.error as e:
            print(f"Display not available: {e}")

        # 응답에 성능 메트릭 포함
        response_data = {
            "status": "received", 
            "detections": detections,
            "performance": {
                "fps": current_fps,
                "bandwidth_bps": current_bandwidth,
                "frame_size_bytes": frame_size,
                "process_time_ms": process_time
            }
        }
        
        return JSONResponse(response_data)
    
    except Exception as e:
        print(f"추론 중 오류 발생: {e}")
        return JSONResponse({"error": f"Inference failed: {str(e)}"}, status_code=500)


@app.get("/")
async def root():
    return {
        "message": "YOLOv8 ONNX Server is running", 
        "model_loaded": ort_session is not None,
        "input_size": input_size if ort_session else None
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy" if ort_session is not None else "model_not_loaded"}


@app.get("/metrics")
async def get_metrics():
    """현재 성능 메트릭 반환"""
    return {
        "fps": perf_tracker.get_fps(),
        "bandwidth_bps": perf_tracker.get_bandwidth(),
        "frame_count": perf_tracker.frame_count
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)