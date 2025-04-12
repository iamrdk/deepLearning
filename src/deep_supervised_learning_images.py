import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np
from tqdm import tqdm


class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.classes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        self.classes.sort()
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        self.image_paths = []
        self.labels = []

        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            class_idx = self.class_to_idx[class_name]

            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    img_path = os.path.join(class_dir, img_name)
                    self.image_paths.append(img_path)
                    self.labels.append(class_idx)

        print(f"Loaded {len(self.image_paths)} images across {len(self.classes)} classes.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


class NeuralNetwork(nn.Module):
    def __init__(self, num_classes):
        super(NeuralNetwork, self).__init__()
        self.model = models.resnet50(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


class Image_Trainer:
    def __init__(self, name, dataset_dir, img_size=256, batch_size=64, val_split=0.2):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
            print(f"Memory Cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")

        self.dataset_dir = dataset_dir
        self.name = name
        self.img_size = img_size
        self.batch_size = batch_size
        self.val_split = val_split

    def process_dataset(self):
        transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        full_dataset = CustomImageDataset(self.dataset_dir, transform)

        self.classes = full_dataset.classes
        self.num_classes = len(self.classes)

        dataset_size = len(full_dataset)
        val_size = int(self.val_split * dataset_size)
        train_size = dataset_size - val_size

        train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

        self.train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2,
                                           pin_memory=torch.cuda.is_available())
        self.val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2,
                                         pin_memory=torch.cuda.is_available())

        print(f"Training samples: {train_size}")
        print(f"Validation samples: {val_size}")
        self.input_size = 3 * self.img_size * self.img_size  # for backward compatibility with saved model info

    def train(self, dataloader, model, loss_fn, optimizer, epoch):
        size = len(dataloader.dataset)
        model.train()
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1} [Train]")
        running_loss = 0.0
        correct = 0

        for batch, (X, y) in enumerate(progress_bar):
            X, y = X.to(self.device), y.to(self.device)
            pred = model(X)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

            progress_bar.set_postfix({
                'loss': f"{running_loss / (batch + 1):.4f}",
                'acc': f"{correct / ((batch + 1) * dataloader.batch_size):.4f}"
            })

        epoch_loss = running_loss / len(dataloader)
        epoch_acc = correct / size
        return epoch_loss, epoch_acc

    def validate(self, dataloader, model, loss_fn):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.eval()
        val_loss, correct = 0, 0

        progress_bar = tqdm(dataloader, desc="[Validate]")

        with torch.no_grad():
            for X, y in progress_bar:
                X, y = X.to(self.device), y.to(self.device)
                pred = model(X)
                val_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

                progress_bar.set_postfix({
                    'loss': f"{val_loss / num_batches:.4f}",
                    'acc': f"{correct / size:.4f}"
                })

        val_loss /= num_batches
        val_acc = correct / size
        print(f"Validation: Accuracy = {(100 * val_acc):>0.1f}%, Avg loss = {val_loss:.4f}")
        return val_loss, val_acc

    def train_and_save(self, epochs=100, learning_rate=1e-4, early_stopping_patience=3):
        model = NeuralNetwork(self.num_classes).to(self.device)
        print(model)

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

        best_val_acc = 0.0
        best_model_weights = None
        early_stopping_counter = 0

        train_losses, train_accs, val_losses, val_accs = [], [], [], []

        for t in range(epochs):
            print(f"\nEpoch {t + 1}/{epochs}\n-------------------------------")
            train_loss, train_acc = self.train(self.train_dataloader, model, loss_fn, optimizer, t)
            val_loss, val_acc = self.validate(self.val_dataloader, model, loss_fn)

            train_losses.append(train_loss)
            train_accs.append(train_acc)
            val_losses.append(val_loss)
            val_accs.append(val_acc)

            scheduler.step(val_loss)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_weights = model.state_dict().copy()
                early_stopping_counter = 0
                print(f"New best model saved with accuracy: {(100 * best_val_acc):>0.1f}%")
            else:
                early_stopping_counter += 1
                print(f"EarlyStopping counter: {early_stopping_counter} out of {early_stopping_patience}")
                if early_stopping_counter >= early_stopping_patience:
                    print("Early stopping triggered")
                    break

        if best_model_weights:
            model.load_state_dict(best_model_weights)
            print(f"Loaded best model weights with val_acc: {(100 * best_val_acc):>0.1f}")

        torch.save(model.state_dict(), f"{self.name}_model.pth")
        torch.save({
            'img_size': self.img_size,
            'input_size': self.input_size,
            'num_classes': self.num_classes,
            'classes': self.classes
        }, f"{self.name}_info.pth")

        print(f"Saved model to {self.name}_model.pth and info to {self.name}_info.pth")
        self.plot_training_metrics(train_losses, val_losses, train_accs, val_accs)
        return model

    def plot_training_metrics(self, train_losses, val_losses, train_accs, val_accs):
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title('Loss Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(train_accs, label='Training Accuracy')
        plt.plot(val_accs, label='Validation Accuracy')
        plt.title('Accuracy Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.savefig(f"{self.name}_training_metrics.png")
        plt.show()

    def plot_predictions(self, num_images=25):
        model_info = torch.load(f"{self.name}_info.pth")
        model = NeuralNetwork(model_info['num_classes'])
        model.load_state_dict(torch.load(f"{self.name}_model.pth"))
        model.to(self.device)
        model.eval()

        dataiter = iter(self.val_dataloader)
        images, labels = next(dataiter)
        images = images[:num_images]
        labels = labels[:num_images]
        images = images.to(self.device)

        with torch.no_grad():
            predictions = model(images).argmax(1).cpu()

        labels = labels.cpu()
        classes = model_info['classes']

        grid_size = int(np.ceil(np.sqrt(num_images)))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
        axes = axes.flatten()

        for i in range(num_images):
            ax = axes[i]
            img = images[i].cpu().permute(1, 2, 0)
            img = img * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
            ax.imshow(img.clamp(0, 1).numpy())
            color = 'green' if predictions[i] == labels[i] else 'red'
            ax.set_title(f"Pred: {classes[predictions[i]]}\nTrue: {classes[labels[i]]}", color=color)
            ax.axis("off")

        for i in range(num_images, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        plt.savefig(f"{self.name}_predictions.png")
        plt.show()

    def inference_on_image(self, image_path):
        model_info = torch.load(f"{self.name}_info.pth")
        img_size = model_info['img_size']
        classes = model_info['classes']

        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(self.device)

        model = NeuralNetwork(model_info['num_classes'])
        model.load_state_dict(torch.load(f"{self.name}_model.pth"))
        model.to(self.device)
        model.eval()

        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)[0]
            _, predicted_idx = torch.max(output, 1)

        predicted_class = classes[predicted_idx.item()]
        probability = probabilities[predicted_idx].item() * 100

        print(f"Predicted class: {predicted_class} with {probability:.2f}% confidence")

        top_probs, top_indices = torch.topk(probabilities, 3)
        print("Top 3 predictions:")
        for i in range(3):
            print(f"  {classes[top_indices[i]]}: {top_probs[i] * 100:.2f}%")

        plt.figure(figsize=(6, 6))
        img = image_tensor[0].cpu().permute(1, 2, 0)
        img = img * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
        plt.imshow(img.clamp(0, 1).numpy())
        plt.title(f"Prediction: {predicted_class} ({probability:.2f}%)")
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    trainer = Image_Trainer(
        name="celebFace",
        dataset_dir="../PyTorch_classification/celebFace",
        batch_size=48)

    trainer.process_dataset()
    model = trainer.train_and_save()
    trainer.plot_predictions()
