import os
import cv2
import numpy as np

dirOfImages = r"C:\Users\User\Python Works\Breast Cancer Segmentation using U-net\dataset\images"
dirOfMasks = r"C:\Users\User\Python Works\Breast Cancer Segmentation using U-net\dataset\masks"
train_images = []
train_masks = []

for img in os.listdir(dirOfImages):
    img = cv2.imread(os.path.join(dirOfImages, img))
    img = cv2.resize(img, (512, 512))
    train_images.append(img)


for img in os.listdir(dirOfMasks):
    img = cv2.imread(os.path.join(dirOfMasks, img))
    img = cv2.resize(img, (512, 512))
    train_masks.append(img)     

images = np.array(train_images)
masks = np.array(train_masks)

print("Images Found: ", len(images))
print("Masks Found: ", len(masks))

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(images, masks, test_size=0.05, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.10, random_state=42)


from PIL import Image

train_img_dest = r"C:\Users\User\Python Works\B-Cancer\data\train_images"
train_mask_dest = r"C:\Users\User\Python Works\B-Cancer\data\train_masks"
val_img_dest = r"C:\Users\User\Python Works\B-Cancer\data\val_images"
val_mask_dest = r"C:\Users\User\Python Works\B-Cancer\data\val_masks"
test_img_dest = r"C:\Users\User\Python Works\B-Cancer\data\test_images"
test_mask_dest = r"C:\Users\User\Python Works\B-Cancer\data\test_masks"

# Saving Training Images

trainCounter = 0
testCounter = 0
valCounter = 0

# Saving all the Training Images and Masks
for img in X_train:
    im = Image.fromarray(img)
    trainCounter += 1
    im.save(os.path.join(train_img_dest, str(trainCounter)+".jpeg"))

trainCounter = 0
    
for img in y_train:
    im = Image.fromarray(img)
    trainCounter += 1
    im.save(os.path.join(train_mask_dest, str(trainCounter)+".jpeg"))
    
    
# Saving all the validation Images and Masks
for img in X_val:
    im = Image.fromarray(img)
    valCounter += 1
    im.save(os.path.join(val_img_dest, str(valCounter)+".jpeg"))

valCounter = 0
    
for img in y_val:
    im = Image.fromarray(img)
    valCounter += 1
    im.save(os.path.join(val_mask_dest, str(valCounter)+".jpeg"))
    
    
# Saving all the Testing Images and Masks
for img in X_test:
    im = Image.fromarray(img)
    testCounter += 1
    im.save(os.path.join(test_img_dest, str(testCounter)+".jpeg"))

testCounter = 0
    
for img in y_test:
    im = Image.fromarray(img)
    testCounter += 1
    im.save(os.path.join(test_mask_dest, str(testCounter)+".jpeg"))