from ultralytics import YOLO
import cv2

# 1. 학습된 모델 로드
model = YOLO("ex.pt") 

# 2. 실시간 카메라 피드
cap = cv2.VideoCapture(0)  # 카메라 연결

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 3. YOLO 모델 추론
    results = model.predict(source=frame, conf=0.7)  # 신뢰도 70% 이상만 탐지(추후 실험 통해 디벨롭)
    annotated_frame = results[0].plot()  # 탐지 결과 시각화

    # 4. 결과 출력
    cv2.imshow("Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
