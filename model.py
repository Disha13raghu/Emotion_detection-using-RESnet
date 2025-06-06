import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout    
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

#analyzing the data

df= pd.read_csv("data/ckextended.csv")
print(df.head())
print(df.columns)
print(df['emotion'])
print(df.shape)
usg=df['Usage']
print(pd.unique(usg))

#normalising data 
import pandas as pd
import numpy as np


IMAGE_WIDTH = 48
IMAGE_HEIGHT = 48
PIXEL_col = 'pixels'


processed_images = []
labels = [] 

for index, row in df.iterrows():
    pixel_string = row[PIXEL_col]

    if pd.isna(pixel_string) or not str(pixel_string).strip():
        continue

    pixels = np.array(str(pixel_string).split(' '), dtype=np.float32) \
    
    if pixels.size != IMAGE_WIDTH * IMAGE_HEIGHT:
      continue

    image_2d = pixels.reshape(IMAGE_HEIGHT, IMAGE_WIDTH)

    # Normalizing pixels to [0,1]range
    normalized_image = image_2d / 255.0

    processed_images.append(normalized_image)
    if 'emotion' in df.columns: 
        labels.append(row['emotion'])
    else:
        labels.append(None) 
        
        
 #splitting data       

# Convert list of images to a single NumPy array
X = np.array(processed_images)
y = np.array(labels) if labels and labels[0] is not None else None

print("shape of x", X.shape)
if y is not None:
    print("shape of labels", y.shape)
    
print(np.unique(y))
print(X.shape[0]  )


NUM_SAMPLES = X.shape[0]
IMAGE_HEIGHT = 48
IMAGE_WIDTH = 48
num_emotions = len(np.unique(y)) 

X_with_channel = np.expand_dims(X, axis=-1)

# converting the grayscale value of X to RGB for Resnet model
X_rgb_input = np.repeat(X_with_channel, 3, axis=-1)

# for every image it puts all values to 0 except the one with correct emotion
y_one_hot = to_categorical(y, num_classes=num_emotions)

#splitting the data
X_train, X_test, y_train, y_test = train_test_split(
    X_rgb_input, y_one_hot,
    test_size=0.2,
    random_state=42,
    stratify=y_one_hot #makes sure the propotion of classification is same in training and testing data for better testing
)  
   
# model with data augumentation and dropout
input_tensor = Input(shape=(48, 48, 3))

#data augument layer 
data_augmentation = tf.keras.Sequential([
    #random layers for better predictions
    tf.keras.layers.RandomFlip("horizontal"), #helps with the orientation change 
    tf.keras.layers.RandomRotation(0.1), #rotate image for model to understand rotated images better
    tf.keras.layers.RandomZoom(0.1), #zoom so that model understand zoomed images too
])

x = data_augmentation(input_tensor) # augmentation applied
#using custom layers 
base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=x) # passing augumented layer

#defining layers 
#base layer
x = base_model.output
#average pooling layer to summarise feature and flattening the image inputs
x = GlobalAveragePooling2D()(x)
#fully connected layer
x = Dense(1024, activation='relu')(x)
# Dropout layer for regularisation
x = Dropout(0.5)(x) 
#last layer uses softmax for prediction probability the most probable value will be output
predictions = Dense(num_emotions, activation='softmax')(x)

model = Model(inputs=input_tensor, outputs=predictions)


# freezing pretrained layers as the model base layers are alreasy trained
for layer in base_model.layers:
    layer.trainable = False
    
    
# callbacks for early stopping and learning rate scheduling 
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10, 
    restore_best_weights=True 
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2, 
    patience=5, 
    min_lr=0.00001, 
    verbose=1
)
callbacks = [early_stopping, reduce_lr]

#  Model Compilation with adam optimizer 
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])       


# Training new layers only phase 1
print("training new layers ")
history = model.fit(
    X_train, y_train,
    epochs=50, 
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=callbacks
) 


# evaluation (phase 1) ---
print("\n--- evaluation after Phase 1 ---")
y_pred_proba_phase1 = model.predict(X_test)
y_pred_phase1 = np.argmax(y_pred_proba_phase1, axis=1)
y_true_test = np.argmax(y_test, axis=1)

print("classification report (phase 1)")
print(classification_report(y_true_test, y_pred_phase1, digits=4))
print(f"accuracy (phase 1): {accuracy_score(y_true_test, y_pred_phase1):.4f}")
print(f"precision (phase 1, weighted): {precision_score(y_true_test, y_pred_phase1, average='weighted'):.4f}")
print(f"recall (phase 1, weighted): {recall_score(y_true_test, y_pred_phase1, average='weighted'):.4f}")
print(f"f1-score (phase 1, weighted): {f1_score(y_true_test, y_pred_phase1, average='weighted'):.4f}")


#finetuning the base model layers 
print("finetuning base model ")
for layer in base_model.layers:
    layer.trainable = True
    
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history_fine_tune = model.fit(
    X_train, y_train,
    epochs=20, 
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=callbacks 
)

# final evaluation
print("final evaluation")
y_pred_proba_final = model.predict(X_test)
y_pred_final = np.argmax(y_pred_proba_final, axis=1)

print("classification report (Final)")
print(classification_report(y_true_test, y_pred_final, digits=4))
print(f"accuracy (final): {accuracy_score(y_true_test, y_pred_final):.4f}")
print(f"precision (final, weighted): {precision_score(y_true_test, y_pred_final, average='weighted'):.4f}")
print(f"recall (final, weighted): {recall_score(y_true_test, y_pred_final, average='weighted'):.4f}")
print(f"f1-score (final, weighted): {f1_score(y_true_test, y_pred_final, average='weighted'):.4f}")    


model.save('emotion_detection_resnet_model.h5')






  
    
    
    
    