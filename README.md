# MN projekt semestralny

## O projekcie
Zebraliśmy zdjęcia oraz informacje o obiektach na nich widocznych, czyli współrzędne przedstawiające położenie obiektów. Podczas oznaczania skupialiśmy się na dwóch typach obiektów: ludziach i samochodach. Ostatecznie modele zostały zaprojektowane tak, by rozpoznawały wyłącznie ludzi. Zebrane dane posłużyły do trenowania modeli, które miały za zadanie wykrywać wybrany obiekt na zdjęciach.

## Folder dataset
https://drive.google.com/drive/folders/1Q753o4HSwhU-J6FX0UaYMG7VkS2Qu8Vp?usp=sharing

## Struktura folderu dataset
dataset/   

│   

├── images/   

│   ├── train/   

│   │   ├── image1.png   

│   │   ├── image2.png   

│   │   └── ...   

│   ├── val/   

│       ├── image341.png   

│       ├── image342.png   

│       └── ...   

│   

├── labels/   

    ├── train/   

    │   ├── image1.txt   

    │   ├── image2.txt   

    │   └── ...   

    ├── val/   

        ├── image341.txt   

        ├── image342.txt   

        └── ...   

## Użyte biblioteki:
PyTorch – Umożliwia implementację modeli sieci neuronowych, przeprowadzanie treningu oraz wykonywanie obliczeń numerycznych na danych.

Torchvision – Zawiera przetrenowane modele, zestawy danych oraz narzędzia do przetwarzania i transformacji obrazów, ułatwiające pracę z danymi wizualnymi.

Matplotlib – Służy do wizualizacji wyników modeli, umożliwia wyświetlanie obrazów oraz oznaczanie wykrytych obiektów na grafice.

PIL (Pillow) – Biblioteka do obsługi obrazów, obejmująca ich wczytywanie, wstępne przetwarzanie oraz konwersję do formatu tensorów kompatybilnych z PyTorch.

Tqdm – Zapewnia czytelne paski postępu podczas wykonywania operacji, takich jak trening modeli lub przetwarzanie dużych zbiorów danych.

Os – Umożliwia operacje na plikach i katalogach, takie jak odnajdywanie plików, odczyt etykiet oraz zarządzanie ścieżkami dostępu.

NumPy – Biblioteka do obliczeń numerycznych, umożliwiająca m.in. przetwarzanie danych, normalizację obrazów oraz konwersję między różnymi formatami danych.

## Kod programu:
'''python
import os
import torch
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torch import nn, optim
from PIL import Image
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

#Klasa do ładowania i przetwarzania zbioru danych obrazów

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, label_dir=None, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.images = os.listdir(image_dir)
        self.transform = transform

    def __len__(self):
	#Zwraca liczbę obrazów w zbiorze danych
        return len(self.images)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.images[index])
        image = Image.open(image_path).convert("RGB")

        label = None
        if self.label_dir:
            label_path = os.path.join(self.label_dir, self.images[index].replace('.png', '.txt'))
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                    labels = [int(line.strip().split()[0]) for line in lines]
                    label = labels[0]
            else:
                print(f"Brak etykiety dla obrazu {self.images[index]}")
	
	#Zastosowanie transformacji
        if self.transform:
            image = self.transform(image)

        return image, label
#Funkcja do trenowania modelu ResNet18
def train_model(train_loader, val_loader, num_classes=2, num_epochs=5, learning_rate=0.001):
    model = models.resnet18(weights='IMAGENET1K_V1')
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
	
	#Pętla po partiach danych
        loop = tqdm(train_loader, leave=True)
        for images, labels in loop:
            images, labels = images.to(device), labels.clone().detach().to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            train_accuracy = 100 * correct_train / total_train
            loop.set_description(f"Epoka [{epoch+1}/{num_epochs}]")
            loop.set_postfix(loss=loss.item(), acc=train_accuracy)

        print(f"Epoka {epoch + 1}/{num_epochs}")
        print(f"  Średnia strata (train): {running_loss / len(train_loader):.4f}")
        print(f"  Dokładność (train): {train_accuracy:.2f}%")
        print("---")

    #Walidacja modelu
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.clone().detach().to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print("\n=== Walidacja trenowanego modelu ===")
    print(f"Dokładność na zbiorze walidacyjnym: {accuracy:.2f}%\n")

    return model

#Funkcja do wizualizacji obrazów z przewidywaniami, etykietami i polami ograniczającymi
def plot_image(images, predictions, labels, confidences, class_names, label_files):
    fig, axs = plt.subplots(1, len(images), figsize=(10, 5))
    for i, ax in enumerate(axs):
        img = images[i].permute(1, 2, 0).cpu().numpy()
        img = (img * 0.229 + 0.485) 
        img = np.clip(img, 0, 1)
        ax.imshow(img)

        pred_class = class_names[predictions[i].item()]
        label_class = class_names[labels[i].item()]
        confidence = confidences[i]

        ax.set_title(
            f"Pred: {pred_class} ({confidence:.2f})\nLabel: {label_class}",
            color="black" if predictions[i] == labels[i] else "red"
        )
        ax.axis("off")

        label_path = label_files[i]
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    try:
                        _, x_center, y_center, width, height = map(float, line.strip().split())
                        img_width, img_height = img.shape[1], img.shape[0]
                        x_center *= img_width
                        y_center *= img_height
                        width *= img_width
                        height *= img_height

                        x_min = x_center - width / 2
                        y_min = y_center - height / 2

                        rect = patches.Rectangle(
                            (x_min, y_min), width, height, linewidth=2, edgecolor='red', facecolor='none'
                        )
                        ax.add_patch(rect)
                    except ValueError as e:
                        print(f"Błąd w odczycie etykiety: {e}")
    plt.tight_layout()
    plt.show()

#Funkcja walidująca model i wyświetlająca poprawne przewidywania
def validate_and_display(val_loader, model, class_names):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    correct_examples = []
    label_files = []
    confidences = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.clone().detach().to(device)
            outputs = model(images)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidences_batch, predicted = torch.max(probabilities, 1)

	    #Zbieranie poprawnych przykładów
            for i, (img, pred, label, confidence) in enumerate(zip(images, predicted, labels, confidences_batch)):
                if pred == label:
                    label_path = val_loader.dataset.label_dir
                    label_file = os.path.join(label_path, val_loader.dataset.images[val_loader.dataset.images.index(os.path.basename(val_loader.dataset.image_dir + '/' + val_loader.dataset.images[i]))].replace('.png', '.txt'))
                    correct_examples.append((img, pred, label))
                    label_files.append(label_file)
                    confidences.append(confidence.item())

                if len(correct_examples) >= 2:
                    break
            if len(correct_examples) >= 2:
                break

    if len(correct_examples) >= 2:
        images, predictions, labels = zip(*correct_examples[:2])
        plot_image(images, predictions, labels, confidences[:2], class_names, label_files[:2])
    else:
        print("Nie znaleziono wystarczającej liczby poprawnych przykładów.")

#Funkcja do walidacji wielu modeli i porównania ich wyników
def validate_model(val_loader, model_names, num_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    class_names = ["Human", "Car"]

    for model_name in model_names:
        print(f"\n=== Walidacja modelu {model_name} ===")
        if model_name == "resnet34":
            model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif model_name == "resnet50":
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif model_name == "vgg16":
            model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
            model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        elif model_name == "densenet":
            model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        elif model_name == "efficientnet_b0":
            model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        else:
            print(f"Model {model_name} nie jest obsługiwany.")
            continue


        model = model.to(device)
        model.eval()
        correct = 0
        total = 0

	#Walidacja na zbiorze walidacyjnym
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.clone().detach().to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        results.append({"model_name": model_name, "accuracy": accuracy})

        print(f"Model: {model_name}")
        print(f"  Dokładność: {accuracy:.2f}%")
        print("---")

    return results

if __name__ == "__main__":
    base_path = '/content/drive/MyDrive/Colab Notebooks/dataset'
    image_train_path = os.path.join(base_path, 'images', 'train')
    image_val_path = os.path.join(base_path, 'images', 'val')
    label_train_path = os.path.join(base_path, 'labels', 'train')
    label_val_path = os.path.join(base_path, 'labels', 'val')

    #Definicja transformacji dla obrazów
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = ImageDataset(image_train_path, label_train_path, transform)
    val_dataset = ImageDataset(image_val_path, label_val_path, transform)

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

    print(f'Liczba obrazów treningowych: {len(train_dataset)}')
    print(f'Liczba obrazów walidacyjnych: {len(val_dataset)}')

    print("\nWalidacja modeli przetrenowanych")
    model_names = ["resnet34", "resnet50", "vgg16", "densenet", "efficientnet_b0"]
    results = validate_model(val_loader, model_names, num_classes=2)

    #Podsumowanie wyników walidacji różnych modeli
    print("\n=== Podsumowanie wyników ===")
    for result in results:
        print(f"Model: {result['model_name']}")
        print(f"  Dokładność: {result['accuracy']:.2f}%")
        print("---")

    #Trenowanie i walidacja modelu ResNet18
    print("\nTrenowanie i walidacja modelu\n")
    print("=== Trening modelu ResNet18 ===")
    trained_model = train_model(train_loader, val_loader, num_classes=2, num_epochs=3)
    
    #Wizualizacja poprawnych predykcji modelu po treningu
    class_names = ["Human", "Car"]
    validate_and_display(val_loader, trained_model, class_names)
'''