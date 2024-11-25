from ultralytics import YOLO

model = YOLO('yolov11n.pt')

results = model.train(
    data="datasets/custom.yaml",  # YAML 파일 경로
    epochs=50,                    # 학습 epoch 수
    imgsz=640,                    # 입력 이미지 크기
    batch=16,                     # 배치 크기
    workers=4,                    # 데이터 로딩 쓰레드 수
    device=0                      # GPU 
)

model.export(format="onnx")