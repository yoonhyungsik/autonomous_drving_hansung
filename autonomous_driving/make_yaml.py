from ultralytics import YOLO
import os
import torch
torch.cuda.empty_cache()


# 사전 학습된 YOLO 모델 로드
model = YOLO('yolov8n.pt')  # COCO 데이터셋으로 학습된 모델

# 라벨 파일 저장 함수
def save_labels(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for result in results:
        print(f"처리 중인 파일: {result.path}")
        file_name = os.path.basename(result.path).replace('.jpg', '.txt')
        label_path = os.path.join(output_dir, file_name)

        with open(label_path, 'w') as f:
            # 탐지된 각 바운딩 박스 정보 저장
            for box in result.boxes:
                cls = int(box.cls.cpu().numpy())  # 클래스 ID (정수)
                x, y, w, h = box.xywhn.cpu().numpy()[0]  # 정규화된 좌표 및 크기
                f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

# 이미지 경로
image_dir = r"C:\Users\ye761\Downloads\autonomous_dataset"
output_label_dir = r"C:\Users\ye761\Downloads\autonomous_dataset\train_val_split\train\images\labels"
images = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".jpg")]

# YOLO 탐지 및 라벨 저장
results = model.predict(source=images, save=False, device=0,stream=True, batch=1, imgsz=640)  # GPU 사용
save_labels(results, output_label_dir)

print(f"라벨 파일이 {output_label_dir}에 저장되었습니다.")
