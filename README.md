# Gender Prediction and Age Group Classification using Deep Learning

This complete AI project uses deep learning to predict the gender and estimate the age group of a person by analyzing their facial features. This is done through a Convolutional Neural Network (CNN) multi-output architecture.

## 📂 Project Structure

- `requirements.txt`: Python dependencies needed to run the project.
- `train_model.py`: Script for loading the dataset, preprocessing it, training the CNN model, and plotting the metrics.
- `train_model.ipynb`: A Jupyter Notebook equivalent that does the same steps with markdown context.
- `app.py`: Streamlit-based Web Application offering a beginner-friendly UI where you can upload pictures and receive predictions.
- `realtime_prediction.py`: An OpenCV real-time script that turns on your webcam and analyzes faces in front of the camera.
- `models/`: Directory where the `.h5` model weights are saved after training.
- `test_images/`: A set of beautiful AI-generated test faces containing Tamil Nadu locals to test the capabilities of your model on Streamlit!
- `UTKFace/UTKFace/`: The unzipped dataset directory (downloaded from Kaggle).

## 🚀 Step-by-Step Explanation

We follow a classic deep learning pipeline:

1. **Data Preprocessing**:
   - We utilize the public **UTKFace** dataset. Images in this dataset are named in a specific format: `[age]_[gender]_[race]_[datetime].jpg`.
   - The code iterates over all images, reads their age and gender from the filename.
   - Ages are converted into **Age Group Categories**: `Child (0-12)`, `Teen (13-19)`, `Young Adult (20-35)`, `Adult (36-55)`, `Senior (56+)`.
   - Images are resized to `64x64` uniformly and loaded as memory-efficient `float32` arrays, normalized `(divided by 255.0)` to aid the neural network's convergence.

2. **Train-Test Split**:
   - Taking 80% of our photos, we use them to let the network *learn* features (training set), and hide the remaining 20% to test its accuracy afterward (testing/validation set).

3. **Model Building (CNN Architecture)**:
   - We design a Convolutional Neural Network (CNN) architecture using TensorFlow/Keras.
   - The network has typical Conv and Pooling blocks extracting facial shapes and edges.
   - **Multi-output**: After flattening, the layer branches into two individual outputs.
     - The first output is a classification branch (`sigmoid`) to guess Gender (Male = 0, Female = 1).
     - The second output is a classification branch (`softmax`) into the 5 Age categories.

4. **Model Training & Evaluation**:
   - The model learns by comparing its guesses against actual correct answers (using loss functions: `binary_crossentropy` for Gender and `sparse_categorical_crossentropy` for Age groups).
   - Once trained, accuracy metrics are plotted to help assess performance.

## ⚙️ How to Run

1. **Enter Project Directory**:
   Open a terminal and execute:
   ```cmd
   cd "c:\xampp\htdocs\task 2\Gender_Age_Prediction"
   ```

2. **Run Live Predictions (Webcam)**:
   Test your model in real time via OpenCV:
   ```cmd
   python realtime_prediction.py
   ```
   *(Press 'q' key to close the webcam window)*

3. **Run Web UI (Bonus UI)**:
   Enjoy a clean web interface using Streamlit:
   ```cmd
   streamlit run app.py
   ```
   *You can securely securely upload images directly from the `test_images/` folder here to test!*
