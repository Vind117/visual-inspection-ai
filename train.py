from ultralytics import YOLO

if __name__ == '__main__':
    # Initialize YOLOv8 Nano pre-trained weights
    model = YOLO('yolov8n.pt')

    # Train for 5 epochs on your custom dataset
    model.train(
        data='dataset.yaml',
        epochs=5,
        imgsz=640,
        batch=16,
        project='runs/train',
        name='industrial_defect_model'
    )
    print("Training complete! Model saved to runs/train/industrial_defect_model/weights/defect video.mp4")