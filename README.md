# plant-seedlings-classification
Overview  
This project focuses on building a deep learning-based image classification model to identify different species of plant seedlings. 
It aims to assist in automated weed detection and crop management, which is useful in smart agriculture.  



PHASE 1 — SETUP & IMPORTS
Tools Used:
* TensorFlow & Keras – To build and train deep learning models
* OpenCV – To process and mask images
* gdown – To download dataset from Google Drive
* NumPy, Pandas, Matplotlib, Seaborn – For data analysis and visualization
Purpose:
 Install required libraries  Import all modules for image preprocessing, model training, and visualization


 PHASE 2 — PARAMETERS
Important project settings:
Parameter	Description
IMG_SIZE = (224, 224)	Image size for model input
BATCH_SIZE = 32	Number of images processed per batch
SEED = 123	Random seed for reproducibility
INITIAL_EPOCHS = 30	Number of epochs for first training phase
FINE_TUNE_EPOCHS = 10	Additional epochs after fine-tuning


Purpose:
 To make the notebook flexible and easy to modify for other datasets.



 PHASE 3 — DATASET DOWNLOAD & EXTRACTION
* The dataset is stored in Google Drive.
* It is downloaded using the gdown library and extracted automatically using zipfile.
Purpose:
 To access and prepare image data for processing and training.



 PHASE 4 — MASKING (BACKGROUND REMOVAL)
Goal:
Remove the non-green background (like soil or stones) and keep only the green seedling part.
Technique Used:

Steps:
1. Convert each image from BGR to HSV color space.
2. Define a green color range: lower = np.array([25, 40, 40])
3. upper = np.array([95, 255, 255])
4. HSV Color Masking using OpenCV
5. Keep only pixels within this range.
6. Save masked (cleaned) images in a new folder.

   
Why HSV?
* HSV separates color (Hue) from brightness (Value), making it easier to detect specific colors under different lighting.

  
Result:
Only plant areas (green pixels) are visible.
 Background (non-green) areas are removed.
Note:
This is not a deep learning model — it’s a classical image processing method using OpenCV.



 PHASE 5 — MASKING VISUALIZATION
* Displays a few “before and after” examples of masking.
* Ensures that the green plant regions are correctly preserved.
Purpose:
 Verify the masking quality visually.



 PHASE 6 — DATASET SUMMARY & BALANCE CHECK
After masking:
* Count how many images exist for each class (category).
* Display a bar chart showing image distribution per class.
Purpose:
 Check for class imbalance (if some classes have much fewer images).  Balanced data helps the model perform better.



 PHASE 7 — PREPARE DATASET FOR TRAINING
Using:
tf.keras.utils.image_dataset_from_directory()
* Splits dataset into 80% training and 20% validation.
* Automatically assigns class labels based on folder names.
* Resizes all images to 224×224.
Purpose:
 Load and preprocess images efficiently for TensorFlow.



 PHASE 8 — DATA AUGMENTATION
Random transformations are applied to make the model more general and robust:
Augmentation	Description
Random Flip	Flips image horizontally
Random Rotation	Rotates image slightly
Random Zoom	Zooms in/out randomly
Random Contrast	Changes brightness contrast
Random Translation	Moves image slightly left/right or up/down
Purpose:
 Increases dataset diversity
 Prevents overfitting



 PHASE 9 — CALLBACKS
Used to control and optimize training:
Callback	Function
EarlyStopping	Stops training when accuracy stops improving
ModelCheckpoint	Saves the best model automatically
ReduceLROnPlateau	Reduces learning rate if model gets stuck
Purpose:
 Avoid overfitting  Improve training efficiency



 PHASE 10 — MODEL CREATION (TRANSFER LEARNING)
Technique:
Transfer Learning using pretrained CNN models from ImageNet.
Code Summary:
base = base_fn(weights='imagenet', include_top=False, input_shape=(224,224,3))
base.trainable = False
x = data_augmentation(inputs)
x = preprocess_fn(x)
x = base(x, training=False)
x = GlobalAveragePooling2D()
x = Dense(256, activation='relu')
output = Dense(num_classes, activation='softmax')
Explanation:
1. Use a pretrained CNN base model (like MobileNetV2 or EfficientNetB0).
2. Freeze its layers initially.
3. Add new layers for your custom classification task.








<img width="1024" height="1536" alt="Hovor queome" src="https://github.com/user-attachments/assets/37067208-bbd2-402c-9b82-37a43d3675d2" />

 
