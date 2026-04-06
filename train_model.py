import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# ==========================================
# 1. Dataset Configuration & Preprocessing
# ==========================================
# This script assumes you have downloaded the UTKFace dataset
# and extracted the images to a folder named 'UTKFace' in the current directory.
DATA_DIR = "UTKFace/UTKFace"
IMG_SIZE = 64

def load_data(data_dir):
    """
    Loads images and extracts age and gender labels from filenames.
    Format: [age]_[gender]_[race]_[datetime].jpg
    Gender: 0 (Male), 1 (Female)
    """
    images = []
    ages = []
    genders = []
    
    if not os.path.exists(data_dir):
        print(f"ERROR: Directory {data_dir} not found.")
        print("Please download the UTKFace dataset and place the extracted images inside a folder named 'UTKFace'.")
        return np.array([]), np.array([]), np.array([])
        
    for filename in os.listdir(data_dir):
        if filename.endswith(".jpg"):
            try:
                # Extract age and gender from filename
                parts = filename.split('_')
                if len(parts) >= 3:
                    raw_age = int(parts[0])
                    gender = int(parts[1])
                    
                    # Convert raw age into age group category (0 to 4) for higher accuracy classification
                    if raw_age <= 12: age = 0
                    elif raw_age <= 19: age = 1
                    elif raw_age <= 35: age = 2
                    elif raw_age <= 55: age = 3
                    else: age = 4
                    
                    # Read and resize image
                    img_path = os.path.join(data_dir, filename)
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                        
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    
                    images.append(img)
                    ages.append(age)
                    genders.append(gender)
            except Exception as e:
                continue
                
    # Normalize images pixel values between 0 and 1
    images = np.array(images, dtype=np.float32) / 255.0
    ages = np.array(ages)
    genders = np.array(genders)
    
    return images, ages, genders

# ==========================================
# 2. Build Multi-Output CNN Model
# ==========================================
def build_model(input_shape=(64, 64, 3)):
    """
    Creates a Convolutional Neural Network with two outputs:
    1. Gender classification (Binary, 0-1)
    2. Age prediction (Regression, Linear output)
    """
    inputs = Input(shape=input_shape)
    
    # Convolutional Blocks
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    # Output Branch 1: Gender (Binary Classification)
    gender_output = Dense(1, activation='sigmoid', name='gender_output')(x)
    
    # Output Branch 2: Age (Classification into 5 categories)
    age_output = Dense(5, activation='softmax', name='age_output')(x)
    
    # Combine outputs into a single model
    model = Model(inputs=inputs, outputs=[gender_output, age_output])
    
    model.compile(
        optimizer='adam',
        loss={
            'gender_output': 'binary_crossentropy',
            'age_output': 'sparse_categorical_crossentropy' # Classification instead of regression
        },
        metrics={
            'gender_output': 'accuracy',
            'age_output': 'accuracy' # Accuracy for Age Groups
        }
    )
    
    return model

# ==========================================
# 3. Main Training Execution
# ==========================================
def main():
    print("Loading data...")
    X, y_age, y_gender = load_data(DATA_DIR)
    
    if len(X) == 0:
        return
        
    print(f"Loaded {len(X)} images.")
    
    # Train-Test Split (80% train, 20% test)
    X_train, X_test, y_gender_train, y_gender_test, y_age_train, y_age_test = train_test_split(
        X, y_gender, y_age, test_size=0.2, random_state=42
    )
    
    print("Building model...")
    model = build_model()
    model.summary()
    
    print("Training model...")
    history = model.fit(
        X_train,
        {'gender_output': y_gender_train, 'age_output': y_age_train},
        validation_data=(X_test, {'gender_output': y_gender_test, 'age_output': y_age_test}),
        epochs=15, # Increased to 15 epochs for maximum accuracyr accuracy is needed
        batch_size=32,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        ]
    )
    
    # Save the trained model
    os.makedirs('models', exist_ok=True)
    model.save('models/gender_age_model.h5')
    print("Model saved to 'models/gender_age_model.h5'")
    
    # ==========================================
    # 4. Evaluation and Plotting
    # ==========================================
    # Plot Gender Accuracy
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['gender_output_accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_gender_output_accuracy'], label='Val Accuracy')
    plt.title('Gender Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot Age Group Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['age_output_accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_age_output_accuracy'], label='Val Accuracy')
    plt.title('Age Group Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

if __name__ == "__main__":
    main()
