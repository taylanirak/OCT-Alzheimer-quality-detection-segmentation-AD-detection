import os
import torch
import random
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset, ConcatDataset, Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score
from PIL import Image
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import copy
from torchvision import models
from torchvision.models import resnet34, ResNet34_Weights



class ApplyCLAHE:
    def __init__(self, clip_limit=3.0, tile_grid_size=(4, 4)): #8,8 de olur, ne kadar az o kadar ayrıntılı   OYNAYABİLİRSİN BUNLAR İLE
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




# Transformation tanımlama
train_transform = transforms.Compose([ #SOR: train datasını, validation ve test dataından daha çok augmente ettiğin için bir nevi %80 %20 lik oranda oynama olmaz mı?(sanki trainde çok daha çeşitli image var gibi)
    ApplyCLAHE(),
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),                                                       
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229]),  # grayscale icin
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3), value='random'),  

])


val_transform = transforms.Compose([
    ApplyCLAHE(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229]),  # grayscale icin
])



class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, split='train', transform=None):
        self.img_labels = pd.read_excel(annotations_file)
        self.img_labels = self.img_labels[self.img_labels['split'] == split]
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 'filename'])
        image = Image.open(img_path).convert("RGB")
        label = int(self.img_labels.iloc[idx, 'label'])
        if self.transform:
            image = self.transform(image)
        return image, label
    


train_dataset = CustomImageDataset(
    annotations_file='path/to/your/excel/file.xlsx',
    img_dir='path/to/images',
    split='train',
    transform=train_transform
)
val_dataset = CustomImageDataset(
    annotations_file='path/to/your/excel/file.xlsx',
    img_dir='path/to/images',
    split='val',
    transform=val_transform
)
test_dataset = CustomImageDataset(
    annotations_file='path/to/your/excel/file.xlsx',
    img_dir='path/to/images',
    split='test',
    transform=val_transform  
)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)                 #num_workers /  pin_memory

#test datasını transformerden geçirip augmente etmeli miyim?

#hypertuning için kullanılacak seçenekler
learning_rates = [0.01, 0.001, 0.0001, 0.00001]
batch_sizes = [32, 64, 128]
optimizers_dict = {
    'adam': lambda params, lr: optim.Adam(params, lr=lr),
    'sgd': lambda params, lr: optim.SGD(params, lr=lr, momentum=0.9)   #BASKA OPTIMIZERLAR DA VAR DENENEBILECEK   (AdamW)
}



best_val_accuracy = 0
best_hyperparams = {}
best_model_state = None




# resnet modeli
class ResNetModel(nn.Module):
    def __init__(self, num_classes=3, dropout_rate=0.5):
        super(ResNetModel, self).__init__()
        weights = ResNet34_Weights.DEFAULT      #ne kadar yüksek model o kadar ezberleme
        original_model = resnet34(weights=weights)
        original_model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        self.features = nn.Sequential(*list(original_model.children())[:-1])
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(original_model.fc.in_features, num_classes)
        )


    def forward(self, x):  
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    
model = ResNetModel(num_classes=3, dropout_rate=0.5).to(device)  



from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

#evaluate model fonksiyonu
def evaluate_model(model, data_loader):
    model.eval()  
    all_labels = []
    all_predictions = []
    all_probs = []
    val_loss_accum = 0.0
    criterion = nn.CrossEntropyLoss()
        
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            val_loss = criterion(outputs, labels)
            val_loss_accum += val_loss.item()
            
            probs = F.softmax(outputs, dim=1).cpu().numpy()  
            predictions = np.argmax(probs, axis=1) 
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions)
            all_probs.extend(probs)


    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average='macro')
    auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')  
      
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }
        
    val_loss_avg = val_loss_accum / len(data_loader)
    print(f'Accuracy: {accuracy*100:.2f}% Precision: {precision:.2f} Recall: {recall:.2f} F1 Score: {f1:.2f} AUC: {auc:.2f}')

    return val_loss_avg, accuracy, metrics



    
#modeli train ve evaluate etme fonskiyonu
def train_and_evaluate_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=1):
    best_metrics = {}
    best_val_accuracy = -float('inf')

    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        val_loss_avg, val_accuracy, current_metrics = evaluate_model(model, val_loader)
        print(f'Epoch {epoch+1}: Val Loss: {val_loss_avg:.4f}, Val Accuracy: {val_accuracy*100:.2f}%')

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_metrics = current_metrics

        scheduler.step(val_loss_avg)

    return best_val_accuracy, best_metrics




#hyperparameter tuning için gereken degiskenlerin yüzdelikleri için fonskiyon
def calculate_weighted_score(auc, val_accuracy, f1_score, recall, precision):
    weights = {'auc': 0.7,   'f1_score': 0.1, 'recall': 0.1, 'precision': 0.1 }
    score = (auc * weights['auc'] +
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
    
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        
        loss = criterion(outputs, targets) 
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)  
        correct_predictions += (preds == targets).sum().item()
        total_predictions += targets.size(0)
    
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
}




#en iyi parametreleri bulma döngüsü
for lr in learning_rates:
    for batch_size in batch_sizes:
        for opt_name, opt_func in optimizers_dict.items():    
            print(f"\nTraining with lr={lr}, batch_size={batch_size}, optimizer={opt_name}")
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            model = ResNetModel(num_classes=3).to(device) 
            optimizer = optimizers_dict[opt_name](model.parameters(), lr=lr)       # L1/L2 regularization denenir
            criterion = nn.CrossEntropyLoss() 
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5)     #farklı schedulerlar da ekleyebilirsin / farklı değerler de deneyebilirsin

            best_val_accuracy, metrics = train_and_evaluate_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                num_epochs=1
            )

            overall_score = calculate_weighted_score(
                auc=metrics['auc'],
                val_accuracy='val_accuracy',
                f1_score=metrics['f1'],
                recall=metrics['recall'],
                precision=metrics['precision']
            )

            if overall_score > best_overall_score:
                best_overall_score = overall_score
                best_hyperparams = {'lr': lr, 'batch_size': batch_size, 'optimizer': opt_name}
                best_model_state = copy.deepcopy(model.state_dict())

print("Best Hyperparameters found based on weighted score:", best_hyperparams)

model.load_state_dict(best_model_state)
torch.save(model.state_dict(), 'best_model_hyperparameters.pth')




#kfold splits
n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

overall_train_losses = []
overall_val_losses = []
overall_train_accuracies = []
overall_val_accuracies = []
fold_results = []

#datasetleri birleştirmek lazım kfold icin
combined_dataset = ConcatDataset([train_dataset, val_dataset])


for fold, (train_idx, val_idx) in enumerate(kfold.split(combined_dataset)):
    print(f'Starting fold {fold + 1}')
    
    
    train_subset = torch.utils.data.Subset(combined_dataset, train_idx)
    val_subset = torch.utils.data.Subset(combined_dataset, val_idx)

    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=64, shuffle=False)

    model = ResNetModel(num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.Adam(model.parameters(), lr=0.000027) #,weight_decay=1e-5
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    best_val_accuracy_fold = 0.0

    best_model_path_fold = f'best_model_fold_{fold+1}.pth'
 
    train_losses_fold = []
    val_losses_fold = []
    train_accuracies_fold = []
    val_accuracies_fold = []
    
    best_val_accuracy_fold = 0.0

    num_epochs = 100
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



val_accuracies = [fr['val_accuracy'] for fr in fold_results]
avg_val_accuracy = np.mean(val_accuracies) if val_accuracies else 0
avg_train_losses = np.mean(overall_train_losses, axis=0)
avg_val_losses = np.mean(overall_val_losses, axis=0)

model.load_state_dict(torch.load('best_model_fold.pth'))  
model.to(device)



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
            _, predicted_classes = torch.max(outputs, dim=1)  
            predictions.extend(predicted_classes.cpu().tolist())
    return predictions




current_accuracy = evaluate_model(model.to(device), val_loader)



#test dataları
test_directory = "C:/Users/tayla/Downloads/test_png"     # bunu güncelleyip aşağıdakinin transformsuz halini koumak lazım ele data gelince
#test_dataset = CustomImageDataset(
#    annotations_file='path/to/your/excel/file.xlsx',
#    img_dir='path/to/images',
#    split='test',
#    transform=val_transform  
#)

#imageları random dagitma



all_files = os.listdir(test_directory)
png_files = [file for file in all_files if file.endswith('.png')]
random.shuffle(png_files)




#model guncelleme fonksiyonu
def update_model(model, image_tensor, label_tensor, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    outputs = model(image_tensor)
    loss = criterion(outputs, label_tensor.to(device)) 
    loss.backward()
    optimizer.step()
    model.eval()
    return loss.item()






def save_results(model, loader, filepath):
    model.eval()
    results = []
    
    with torch.no_grad():
        for images, paths in loader:
            images = images.to(device)
            outputs = model(images)
            probabilities = F.softmax(outputs, dim=1).cpu().numpy()  
            for path, probs in zip(paths, probabilities):
                results.append([os.path.basename(path), *probs])
    
    
    columns = ['image_name', 'Bad', 'Usable', 'Good']
    

    df = pd.DataFrame(results, columns=columns)
    df.to_csv(filepath, index=False)
    print(f"Results saved to {filepath}")

output_csv_path = 'your_output_file.csv'
save_results(model, test_directory, output_csv_path)



#grad cam oncesi hazirlik
avg_train_losses = np.mean(overall_train_losses, axis=0)
avg_val_losses = np.mean(overall_val_losses, axis=0)
avg_train_accuracies = np.mean(overall_train_accuracies, axis=0)
avg_val_accuracies = np.mean(overall_val_accuracies, axis=0)



x=0
if x==0:

    #grad_cam uygulama  
    def apply_grad_cam(model, images, target_layer):                        #LIME ya da SHAP da dene
        gradients = []
        activations = []

        def backward_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0])

        def forward_hook(module, input, output):
            activations.append(output)

        handle_forward = target_layer.register_forward_hook(forward_hook)
        handle_backward = target_layer.register_full_backward_hook(backward_hook)

        outputs = model(images)
        model.zero_grad()

        
        probabilities = F.softmax(outputs, dim=1)
        class_idx = probabilities.argmax(dim=1).item()
        score = outputs[:, class_idx]
        score.backward()

        gradients = gradients[0].detach()
        activations = activations[0].detach()
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

        for i in range(activations.shape[1]):
            activations[:, i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(activations, dim=1).squeeze()
        heatmap = F.relu(heatmap)
        heatmap /= torch.max(heatmap)

        handle_forward.remove()
        handle_backward.remove()

        return heatmap.cpu().numpy()
        


    #overlay heatmap fonksiyonu
    def overlay_heatmap(image, heatmap, alpha=0.6):
        image_np = np.array(image)
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.resize(heatmap, (image_np.shape[1], image_np.shape[0]))
        superimposed_img = cv2.addWeighted(image_np, alpha, heatmap, 1 - alpha, 0)
        superimposed_img_pil = Image.fromarray(superimposed_img)
        return superimposed_img_pil


    images, _ = next(iter(val_loader))  

    image_pil = transforms.ToPILImage()(images[0]).convert("RGB")  
    image_tensor = val_transform(image_pil).unsqueeze(0).to(device) 

    model.eval()
    outputs = model(image_tensor)
    predicted_prob = torch.sigmoid(outputs)
    threshold = 0.5  
    predicted_class = (predicted_prob > threshold).int()  
    predicted_quality = 'good' if predicted_class.item() == 1 else 'bad'

    heatmap = apply_grad_cam(model, image_tensor, model.features[4])

    image_np = images[0].cpu().numpy().transpose(1, 2, 0)
    image_np = (image_np * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
    image_np = np.clip(image_np, 0, 1)  

    image_np_uint8 = np.uint8(255 * image_np) 
    superimposed_img = overlay_heatmap(image_np_uint8, heatmap)




    def process_and_visualize_grad_cam(image_path, model, val_transform, target_layer):
        image_pil = Image.open(image_path).convert("RGB")
        image_tensor = val_transform(image_pil).unsqueeze(0).to(device)

        model.eval()
        heatmap = apply_grad_cam(model, image_tensor, target_layer)
        superimposed_img = overlay_heatmap(image_pil, heatmap)

        
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(image_pil)
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(superimposed_img)
        plt.title("Grad-CAM")
        plt.axis('off')
        plt.show()




    # test imageları
    png_files = [file for file in os.listdir(test_dataset) if file.endswith('.png')]


    # hedef layer gostermek grad cam icin
    model = ResNetModel().to(device)
    model.load_state_dict(torch.load('model_complete.pth'))
    target_layer = model.features[4]  




    # her imagea grad cam uygulanması icin gereken dongu
    for file_name in png_files:
        image_path = os.path.join(test_directory, file_name)
        process_and_visualize_grad_cam(image_path, model, val_transform, target_layer)




    #grafikler
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(len(avg_train_losses)), avg_train_losses, label='Average Training Loss')
    plt.plot(range(len(avg_val_losses)), avg_val_losses, label='Average Validation Loss')
    plt.title('Training and Validation Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()


    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 2)
    plt.plot(range(len(avg_train_accuracies)),avg_train_accuracies, label='Training Accuracy')
    plt.plot(range(len(avg_val_accuracies)),avg_val_accuracies, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
