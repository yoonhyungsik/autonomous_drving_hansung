from ultralytics import YOLO

import os
from ultralytics import YOLO
import cv2

# YOLO 모델 로드
model = YOLO('yolov8n.pt')  # YOLOv8 네트워크 모델 (yolov8n.pt는 경량 모델입니다)

# 이미지 파일이 있는 디렉토리 경로
image_directory = r"C:\Users\ye761\Downloads"  # 이미지가 있는 디렉토리 경로

# 이미지 파일 목록 가져오기 (.jpg 파일)
image_files = [f for f in os.listdir(image_directory) if f.endswith('.jpg')]

# 각 이미지를 순차적으로 처리하여 감지
for image_file in image_files:
    image_path = os.path.join(image_directory, image_file)
    
    # 이미지 로드
    image = cv2.imread(image_path)
    
    # 모델을 사용하여 감지
    results = model(image)

    # 감지된 결과 표시
    results[0].show()  # 감지된 이미지 표시
    break
    '''
    # 감지된 이미지를 파일로 저장 (선택 사항)
    output_path = os.path.join(image_directory, f"detected_{image_file}")
    results.save(path=output_path)  # 결과를 저장
    '''
