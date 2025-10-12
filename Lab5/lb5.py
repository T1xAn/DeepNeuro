import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используется устройство: {device}")

MODEL_PATH = "food_classifier.pth"

def setup_model(num_classes=3):

    model = models.vgg11(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False

    num_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_features, num_classes)
    
    model = model.to(device)
    
    if os.path.exists(MODEL_PATH):
        print("Найден файл модели. Загружаем веса...")
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            print("Модель загружена")
            need_training = False
        except Exception as e:
            print(f"Ошибка при загрузке модели: {e}")
            print("Будет выполнено обучение с нуля")
            need_training = True
    else:
        print("Файл модели не найден. Будет выполнено обучение с нуля")
        need_training = True
    
    return model, need_training


def save_model(model, path=MODEL_PATH):
    torch.save(model.state_dict(), path)

# Преобразования для данных
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

try:
    train_dataset = datasets.ImageFolder(
        root='custom_dataset/train',
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        root='custom_dataset/val',
        transform=val_transform
    )

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"Классы: {train_dataset.classes}")
    print(f"Размер тренировочной выборки: {len(train_dataset)}")
    print(f"Размер валидационной выборки: {len(val_dataset)}")

except Exception as e:
    print(f"Ошибка при загрузке данных: {e}")
    exit()

model, need_training = setup_model(num_classes=3)

def calculate_accuracy(outputs, labels):
    _, predicted = torch.max(outputs.data, 1)
    correct = (predicted == labels).sum().item()
    return correct / labels.size(0)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    best_val_accuracy = 0.0
    
    for epoch in range(num_epochs):

        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(torch.max(outputs, 1)[1] == labels.data)
        
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)
        
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_running_loss += loss.item() * inputs.size(0)
                val_running_corrects += torch.sum(torch.max(outputs, 1)[1] == labels.data)
        
        val_epoch_loss = val_running_loss / len(val_dataset)
        val_epoch_acc = val_running_corrects.double() / len(val_dataset)
        
        train_losses.append(epoch_loss)
        val_losses.append(val_epoch_loss)
        train_accuracies.append(epoch_acc.cpu().numpy())
        val_accuracies.append(val_epoch_acc.cpu().numpy())
        
        if val_epoch_acc > best_val_accuracy:
            best_val_accuracy = val_epoch_acc
            save_model(model)
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}')
        print(f'Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}')
        print('-' * 50)
    
    return train_losses, val_losses, train_accuracies, val_accuracies

if need_training:
    print("Начало обучения...")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier[6].parameters(), lr=0.001)
    
    train_losses, val_losses, train_accuracies, val_accuracies = train_model(
        model, train_loader, val_loader, criterion, optimizer, num_epochs=10
    )
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Функция потерь')
    plt.xlabel('Эпоха')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.title('Точность')
    plt.xlabel('Эпоха')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
else:
    print("Обучение не требуется, модель уже загружена")
    
    model.eval()
    val_running_corrects = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            val_running_corrects += torch.sum(torch.max(outputs, 1)[1] == labels.data)
    
    val_accuracy = val_running_corrects.double() / len(val_dataset)
    print(f"Точность загруженной модели на валидационной выборке: {val_accuracy:.4f}")

def predict_images_in_folder(folder_path, model, transform, class_names, device, max_images_per_page=6):

    model.eval()
    
    image_extensions = ['.jpg', '.jpeg']
    image_files = [f for f in os.listdir(folder_path) 
                  if os.path.isfile(os.path.join(folder_path, f)) and 
                  os.path.splitext(f)[1].lower() in image_extensions]
    
    if not image_files:
        print(f"В папке {folder_path} не найдено изображений")
        return
    
    print(f"Найдено {len(image_files)} изображений для обработки")
    
    all_predictions = []
    
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(image_tensor)
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
                predicted_class = torch.argmax(probabilities).item()
                confidence = probabilities[predicted_class].item()
            
            all_predictions.append({
                'file': image_file,
                'image': image,  # Сохраняем PIL Image
                'predicted_class': predicted_class,
                'class_name': class_names[predicted_class],
                'confidence': confidence
            })
            
            print(f"✅ {image_file}: {class_names[predicted_class]} ({confidence:.2%})")
            
        except Exception as e:
            print(f"❌ Ошибка при обработке файла {image_file}: {e}")
    
    total_pages = (len(all_predictions) + max_images_per_page - 1) // max_images_per_page
    
    for page in range(total_pages):
        start_idx = page * max_images_per_page
        end_idx = min((page + 1) * max_images_per_page, len(all_predictions))
        page_predictions = all_predictions[start_idx:end_idx]
        
        ncols = 3
        nrows = (len(page_predictions) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5 * nrows))
        
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])
        
        axes_flat = axes.flatten()
        
        for idx, pred in enumerate(page_predictions):
            ax = axes_flat[idx]
            
            ax.imshow(np.array(pred['image']))
            ax.set_title(f"{pred['class_name']}\n(вероятность: {pred['confidence']:.2%})", 
                        fontsize=10, pad=10)
            ax.axis('off')
        
        for idx in range(len(page_predictions), len(axes_flat)):
            axes_flat[idx].axis('off')
        
        plt.tight_layout()
        plt.suptitle(f"Результаты классификации (страница {page + 1}/{total_pages})", 
                    y=1.02, fontsize=14)
        plt.show()

class_names = train_dataset.classes
print(f"\nКлассы модели: {class_names}")

test_folder = "Food_Data/testings"
    
predict_images_in_folder(test_folder, model, val_transform, class_names, device)