from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2
import matplotlib.pyplot as plt

image_path = '/cta/users/UK_Biobank/Taylan/AD_NonAD_Data/train/CN/5440099_21011_0_0_64_r.png'
mask_path = '/cta/users/UK_Biobank/Taylan/AD_NonAD_Data/train_masks/CN/5440099_21011_0_0_64_r.png'

# image_path = '/cta/users/UK_Biobank/Taylan/AD_NonAD_Data/train/CN/4344647_21011_0_0_64_r.png'
# mask_path = '/cta/users/UK_Biobank/Taylan/AD_NonAD_Data/train_masks/CN/4344647_21011_0_0_64_r.png'

rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]
rgb_img = cv2.resize(rgb_img, (256, 256))
rgb_img = np.float32(rgb_img) / 255

image = read_image(image_path).float() / 255  # Convert to float and scale
mask = read_image(mask_path).float() / 255  # Convert to float and scale

# Resize images and masks to 256x256
image = transforms.Resize((256, 256))(image)
mask = transforms.Resize((256, 256))(mask)

# Convert mask to single channel if necessary
if mask.shape[0] > 1:
    mask = mask[0].unsqueeze(0)  # Assuming the relevant data is in the first channel

# Concatenate image and mask along channel dimension
combined_img = torch.cat([image, mask], dim=0)

combined_img = val_transform(combined_img)

input_tensor = combined_img.unsqueeze(0).to(device)

target_layers = [model.base_model.layer1[-1]]
cam = GradCAMPlusPlus(model=model, target_layers=target_layers)

targets = [ClassifierOutputTarget(1)]

# You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

# In this example grayscale_cam has only one image in the batch:
grayscale_cam = grayscale_cam[0, :]
visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

print (cam.outputs)

_, preds = torch.max(cam.outputs, 1)
print (preds)

plt.imshow(visualization)
plt.show()