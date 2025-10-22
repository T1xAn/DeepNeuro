import os
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
from ultralytics import YOLO

MODEL_PATH = 'yolov8n.pt'
DATA_YAML = 'CatsDataset/dataset.yaml'
OUTPUT_DIR = 'detection_results'

def train_model():
    if os.path.exists(MODEL_PATH):
        print("Модель найдена, загрузка...")
        return YOLO(MODEL_PATH)
    else:
        print("\n Обучение модели...")
        model = YOLO('yolov8n.pt')
        model.train(
            data=DATA_YAML,
            epochs=10,
            imgsz=640,
            device='cpu'
        )
        model.save(MODEL_PATH)
        print(f"\nМодель сохранена: {MODEL_PATH}")
        return model

def plot_detection(image_path, results, save_path):

    image = Image.open(image_path)
    img_array = np.array(image)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(img_array)
    
    for r in results:
        boxes = r.boxes
        if boxes is not None:
            for box in boxes:
                
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].item()
                
                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                    linewidth=3, edgecolor='red', facecolor='none')
                ax.add_patch(rect)
                
                ax.text(x1, y1-10, f'cat {conf:.2f}', 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='red', alpha=0.7),
                    fontsize=12, color='white', fontweight='bold')
    
    ax.set_title(f'Детекция: {os.path.basename(image_path)}', fontsize=14)
    ax.axis('off')
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"Сохранённые результаты: {save_path}")
    
    plt.show()
    plt.close()

def detect_images(model, image_folder, conf=0.5):

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_paths.extend(glob.glob(os.path.join(image_folder, ext)))
    
    print(f"🔍 Найдено {len(image_paths)} изображений")
    
    for img_path in image_paths:
        print(f"\n--- Обработка: {os.path.basename(img_path)} ---")
        
        results = model.predict(img_path, conf=conf, save=True)
        
        for r in results:
            if len(r.boxes) > 0:
                print(f"Найдено котов: {len(r.boxes)}")
                for i, box in enumerate(r.boxes):
                    conf = box.conf[0].item()
                    print(f"   {i+1}. уверенность: {conf:.3f}")
            else:
                print("Коты не найдены")
        
        plot_save_path = os.path.join(OUTPUT_DIR, f"plot_{os.path.basename(img_path)}.png")
        plot_detection(img_path, results, plot_save_path)

if __name__ == "__main__":

    model = train_model()
    detect_images(model, 'test_images')