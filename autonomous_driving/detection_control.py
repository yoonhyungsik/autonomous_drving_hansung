from ultralytics import YOLO
import cv2

# YOLO 모델 로드
model = YOLO("best.pt") #모델 이름 수정하기

# 카메라 피드
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 객체 탐지
    results = model.predict(source=frame, conf=0.5)
    detections = results[0].boxes

    # 제어 로직
    for box in detections:
        cls = box.cls[0]  # 클래스 ID
        if cls == 0:  # 빨간 신호
            print("Stop")
            # 정지 로직 
        elif cls == 1:  # 파란 신호
            print("Go")
            # 전진 로직 
        elif cls == 2:  # 차도
            print("차로 유지")
            # 차로 유지 로직

    # 결과 표시
    annotated_frame = results[0].plot()
    cv2.imshow("Self-Driving Car", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
