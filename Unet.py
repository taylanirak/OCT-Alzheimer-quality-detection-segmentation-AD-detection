import os
import torch
import random
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset, ConcatDataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score
from PIL import Image
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import cv2
import copy
from torchvision import models
from torchvision.models import ResNeXt101_64X4D_Weights , resnext101_64x4d
from torch.utils.data import Dataset


def P_search(x):
    x = 0
    if x==0:
        return False


class ApplyCLAHE:
    def __init__(self, clip_limit=3.0, tile_grid_size=(4, 4)): #8,8 de olur, ne kadar az o kadar ayrıntılı
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    
    def __call__(self, img):
    
        img_np = np.array(img)
        if len(img_np.shape) == 2 or img_np.shape[2] == 1:  # Grayscale
            img_np = self.clahe.apply(img_np)
        else:  # RGB
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            img_np = self.clahe.apply(img_np)
        
        img = Image.fromarray(img_np)
        return img



#gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


from torchvision.transforms import functional as TF

import torch
import kornia.augmentation as K
import torchvision.transforms as transforms




class ElasticDeformation(torch.nn.Module):
    """Apply elastic deformation on image tensor."""
    def __init__(self, alpha=1, sigma=0.05, p=0.3):
        super().__init__()
        self.p = p
        self.elastic = K.RandomElasticTransform(alpha=(alpha, alpha), sigma=(sigma, sigma), p=1.0, align_corners=True)

    def forward(self, img):
        if torch.rand(1) < self.p:
            return self.elastic(img)
        return img





from torchvision import transforms
import torch

# Image transformations for training
train_image_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),
    transforms.ToTensor(),  # Converts PIL Image to Tensor
    ElasticDeformation(alpha=34, sigma=4, p=0.5),
    transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
    transforms.RandomApply([transforms.GaussianBlur(3), transforms.Lambda(lambda x: torch.randn_like(x) * 0.1 + x)], p=0.3),
    transforms.Normalize(mean=[0.485], std=[0.229])  # Adjusted for single-channel

])


# Mask transformations for training (only spatial transformations)
train_mask_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),
    ElasticDeformation(alpha=34, sigma=4, p=0.5),  # Only if you want deformation on masks
    transforms.ToTensor()
])

# Image transformations for validation
val_image_transform = transforms.Compose([
    ApplyCLAHE(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Mask transformations for validation (only resizing and cropping, no color transformations)
val_mask_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])


# Example setup for grayscale images or masks
grayscale_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # For three-channel RGB images
])

# Example setup for RGB images
rgb_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # For three-channel RGB images
])



from PIL import Image

import os
from PIL import Image
from torch.utils.data import Dataset

import os
from PIL import Image
from torch.utils.data import Dataset

from torchvision import transforms
from PIL import Image

from torchvision import transforms
from PIL import Image

class MedicalImageDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_transform=None, mask_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.images = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
        self.masks = [f for f in os.listdir(mask_dir) if os.path.isfile(os.path.join(mask_dir, f))]
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        image = Image.open(img_path).convert("RGB")  # Converts to RGB if not already

        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        mask = Image.open(mask_path).convert("L")  # Masks are grayscale

        if self.image_transform:
            image = self.image_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask

# Example transformations
train_image_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),  # Convert image to tensor before applying further transformations
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_mask_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),  # Convert mask to tensor before applying further transformations
])













# datasetleri loadlama
train_dataset = MedicalImageDataset(
    image_dir='C:/Users/tayla/OneDrive/Masaüstü/AD_NonAD_Data/train (2)/',
    mask_dir='C:/Users/tayla/OneDrive/Masaüstü/AD_NonAD_Data/masked_train (2)/',
    image_transform=train_image_transform,
    mask_transform=train_mask_transform
)

# Create instances of the dataset for validation
val_dataset = MedicalImageDataset(
    image_dir='C:/Users/tayla/OneDrive/Masaüstü/AD_NonAD_Data/val/',
    mask_dir='C:/Users/tayla/OneDrive/Masaüstü/AD_NonAD_Data/masked_val/',
    image_transform=val_image_transform,
    mask_transform=val_mask_transform
)


train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)


#hypertuning için kullanılacak seçenekler
learning_rates = [0.01, 0.001, 0.0001, 0.00001]
batch_sizes = [32, 64, 128]
optimizers_dict = {
    'adam': lambda params, lr: optim.Adam(params, lr=lr),
    'sgd': lambda params, lr: optim.SGD(params, lr=lr, momentum=0.9)
}



best_val_accuracy = 0
best_hyperparams = {}
best_model_state = None

from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from torch import nn
import torch.nn.functional as F





class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1) 

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.double_conv(x)
        return x + residual






class Down(nn.Module):  
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        return self.maxpool_conv(x)





class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)





class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)




class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits




#dice score fonksiyonu
class DiceBCELoss(nn.Module):
    def __init__(self, weight_dice=1.0, weight_bce=1.0):
        super().__init__()
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.weight_dice = weight_dice
        self.weight_bce = weight_bce

    def forward(self, inputs, targets):
        dice_loss = self.dice_loss(inputs, targets)
        bce_loss = self.bce_loss(inputs, targets)
        return self.weight_dice * dice_loss + self.weight_bce * bce_loss

class DiceLoss(nn.Module):
    def forward(self, logits, true):
        eps = 1e-7
        sigmoid = torch.sigmoid(logits)
        smooth = 1.0

        intersection = (sigmoid * true).sum()
        union = sigmoid.sum() + true.sum()
        
        dice = (2. * intersection + smooth) / (union + smooth)
        return 1 - dice




#evaluate model fonksiyonu
def evaluate_model(model, data_loader):
    model.eval()  
    all_labels = []
    all_predicted = []
    all_probs = []
    val_loss_accum = 0.0
        
    with torch.no_grad():
        for data in data_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images).squeeze()
            val_loss = criterion(outputs, labels.float()) 
            val_loss_accum += val_loss.item()  
            
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_labels.extend(labels.tolist())
            all_probs.extend(probs)

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_predicted = (all_probs > 0.5)

    assert all_labels.shape == all_predicted.shape, "Shape mismatch between labels and predictions"

    val_accuracy = 100 * np.mean((all_probs > 0.5) == all_labels)

    accuracy = 100 * np.sum(all_labels == all_predicted) / len(all_labels)
    precision = precision_score(all_labels, all_predicted,zero_division=0)
    recall = recall_score(all_labels, all_predicted,zero_division=0)
    f1 = f1_score(all_labels, all_predicted)
    auc = roc_auc_score(all_labels, all_probs) 
    dice = dice_score(all_labels, all_predicted)  
        
    val_loss_avg = val_loss_accum / len(data_loader)

    metrics = {
            'val_loss_avg': val_loss_avg,
            'val_accuracy': val_accuracy,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'dice_score': dice,
        }
    print(f'Accuracy: {accuracy}%')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')
    print(f'AUC: {auc}') 
    print(f'Dice Score: {dice}')
        
    return val_loss_avg, val_accuracy, metrics



    
#modeli train ve evaluate etme fonskiyonu
def train_and_evaluate_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=50):
    best_metrics = {}
    best_val_accuracy = -float('inf')

    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()

     
        val_loss_avg, val_accuracy, current_metrics = evaluate_model(model, val_loader)

        print(f'Epoch {epoch+1}: done, Val Loss: {val_loss_avg}, Val Accuracy: {val_accuracy}%')

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_metrics = current_metrics

        scheduler.step(val_loss_avg)

    return best_val_accuracy, best_metrics




#hyperparameter tuning için gereken degiskenlerin yüzdelikleri için fonskiyon
def calculate_weighted_score(auc, val_accuracy, dice_score, f1_score, recall, precision):
    weights = {'auc': 0.3, 'val_accuracy': 0.2, 'dice_score': 0.2, 'f1_score': 0.1, 'recall': 0.1, 'precision': 0.1 }
    score = (auc * weights['auc'] +
             val_accuracy * weights['val_accuracy'] +
             dice_score * weights['dice_score'] +
             precision * weights["precision"]+
             f1_score * weights['f1_score'] +
             recall * weights['recall'])
    return score




#bundan yukarıda bir tane daha var hangisini kullanacagına bakarsın
best_overall_score = -float('inf')
best_hyperparams = {}
best_model_state = None




#tek bir epoch icin
def train_one_epoch(epoch, model, train_loader, device, optimizer, criterion):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
        
    for inputs, masks in train_loader:
        inputs, masks = inputs.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
            
        loss = criterion(outputs.squeeze(), masks.float())
        loss.backward()
        optimizer.step()
            
        running_loss += loss.item()
        preds = torch.sigmoid(outputs) > 0.5
        correct_predictions += (preds.squeeze().long() == masks).sum().item()
        total_predictions += masks.size(0)
        
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct_predictions / total_predictions
    print(f'Epoch [{epoch+1}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')
        
    return epoch_loss, epoch_acc


metrics_history = {
    'auc': [],
    'precision': [],
    'accuracy': [],
    'recall': [],
    'f1': [],
    'dice_score': [],
}



if P_search == True:
    #en iyi parametreleri bulma döngüsü
    for lr in learning_rates:
        for batch_size in batch_sizes:
            for opt_name, opt_func in optimizers_dict.items():
                print(f"\nTraining with lr={lr}, batch_size={batch_size}, optimizer={opt_name}")
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

                model = UNet(n_channels=1, n_classes=1).to(device)
                optimizer = optimizers_dict[opt_name](model.parameters(), lr=lr)
                criterion = nn.BCEWithLogitsLoss()
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5)

                best_val_accuracy, metrics = train_and_evaluate_model(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    criterion=criterion,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    device=device,
                    num_epochs=50
                )

                overall_score = calculate_weighted_score(
                    auc=metrics['auc'],
                    val_accuracy=metrics['val_accuracy'],  
                    dice_score=metrics['dice_score'],
                    f1_score=metrics['f1'],
                    recall=metrics['recall'],
                    precision=metrics['precision']
                )

                if overall_score > best_overall_score:
                    best_overall_score = overall_score
                    best_hyperparams = {'lr': lr, 'batch_size': batch_size, 'optimizer': opt_name}
                    best_model_state = copy.deepcopy(model.state_dict())

    print("Best Hyperparameters found based on weighted score:", best_hyperparams)

    # hyperparametreler için en iyi modeli yükle 
    model.load_state_dict(best_model_state)
    torch.save(model.state_dict(), 'best_model_hyperparameters.pth')




#kfold splits
n_splits = 2
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

overall_train_losses = []
overall_val_losses = []
overall_train_accuracies = []
overall_val_accuracies = []
fold_results = []

#datasetleri birleştirmek lazım kfold icin
combined_dataset = ConcatDataset([train_dataset, val_dataset])




for fold, (train_idx, val_idx) in enumerate(kfold.split(np.arange(len(combined_dataset)))):
    print(f'Starting fold {fold + 1}')
    
    train_subset = Subset(combined_dataset, train_idx)
    val_subset = Subset(combined_dataset, val_idx)
    
    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True) # 32den çok daha stabil ve yakın sonuçlar getiriyo  accuracy genelde 72
    val_loader = DataLoader(val_subset, batch_size=64, shuffle=False)

    model = UNet(n_channels=1, n_classes=1).to(device)
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = optim.Adam(model.parameters(), lr=0.000027) #,weight_decay=1e-5
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    best_val_accuracy_fold = 0.0

    best_model_path_fold = f'best_model_fold_{fold+1}.pth'
 
    train_losses_fold = []
    val_losses_fold = []
    train_accuracies_fold = []
    val_accuracies_fold = []
    
    best_val_accuracy_fold = 0.0

    num_epochs = 1
    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(epoch, model, train_loader, device, optimizer, criterion)
        
        train_losses_fold.append(train_loss)
        train_accuracies_fold.append(train_acc)
        
        val_loss_avg, val_accuracy, metrics = evaluate_model(model, val_loader)
        
        for key in metrics_history.keys():
            metrics_history[key].append(metrics[key.replace('val_', '')])

        val_losses_fold.append(val_loss_avg)
        val_accuracies_fold.append(val_accuracy)
        
        if val_accuracy > best_val_accuracy_fold:
            best_val_accuracy_fold = val_accuracy
            torch.save(model.state_dict(), best_model_path_fold)
        
        scheduler.step(val_loss_avg)
    
    
    fold_results.append({
        'fold': fold + 1,
        'train_loss': train_losses_fold,
        'val_loss': val_losses_fold,
        'train_accuracy': train_accuracies_fold,
        'val_accuracy': val_accuracies_fold,
        'best_val_accuracy': best_val_accuracy_fold
    })
    print(f'Finished fold {fold + 1}/{n_splits}')

    torch.save(model, 'model_complete.pth')



val_accuracies = [fr['val_accuracy'] for fr in fold_results]
avg_val_accuracy = np.mean(val_accuracies) if val_accuracies else 0
avg_train_losses = np.mean(overall_train_losses, axis=0)
avg_val_losses = np.mean(overall_val_losses, axis=0)



model = torch.load('model_complete.pth')
model = model.to(device)  

max_values = {metric: max(values) for metric, values in metrics_history.items()}
print("Maximum values for each metric:", max_values)




#predict batch fonskiyonu
def predict_batch(model, data_loader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for images, _ in data_loader: 
            images = images.to(device)
            outputs = model(images)
            predicted_probs = torch.sigmoid(outputs)
            predicted_classes = (predicted_probs > 0.5).int()
            predictions.extend(predicted_classes.squeeze().tolist())
    return predictions




current_accuracy = evaluate_model(model.to(device), val_loader)


#test dataları
image_directory = "C:/Users/Taylan/Downloads/testAD/testAD"

#imageları random dagitma
all_files = os.listdir(image_directory)
png_files = [file for file in all_files if file.endswith('.png')]
random.shuffle(png_files)




#model guncelleme fonksiyonu
def update_model(model, image_tensor, label_tensor):
    model.train()
    optimizer.zero_grad()
    outputs = model(image_tensor)
    loss = criterion(outputs, label_tensor.unsqueeze(1).float().to(device)) 
    loss.backward()
    optimizer.step()
    model.eval()
    return loss.item()




results_file_path = "alz.txt"
with open(results_file_path, 'w') as results_file:
    for file_name in png_files:
        image_path = os.path.join(image_directory, file_name)
        image = Image.open(image_path).convert('RGB')
        transformed_image = val_transform(image).unsqueeze(0).to(device)

        with torch.no_grad(): 
            outputs = model(transformed_image)
            predicted_prob = torch.sigmoid(outputs) 
            threshold = 0.5  
            predicted_class = (predicted_prob > threshold).int()  
            predicted_quality = 'AD' if predicted_class.item() == 1 else 'nonAD'
            results_file.write(f'Image: {file_name}, Predicted quality: {predicted_quality}\n')

torch.save(model.state_dict(), 'model_complete.pth')
print('Finished processing all images.') 