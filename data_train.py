from ultralytics import YOLO

if __name__ == "__main__":
    # YOLOv8 Nano 모델 불러오기
    model = YOLO("yolov8n.pt")

    # 학습 실행 (GPU 0번 사용)
    model.train(
        data="./dataset/data.yaml",
        epochs=50,
        imgsz=640,
        device=0
    )
