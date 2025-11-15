🌱 Waste Classification using CNN

AI-powered waste detection using Convolutional Neural Networks (CNNs)
Promoting smart waste management for a cleaner, sustainable environment.

📌 Overview

This project uses a Convolutional Neural Network (CNN) to automatically classify waste images into categories such as Organic, Recyclable, and Hazardous.
The goal is to support smart cities, IoT dustbins, and environment monitoring systems through automated waste segregation.

The project is implemented using TensorFlow/Keras and can be run on Google Colab or VS Code.

✨ Features

🚀 End-to-end Deep Learning pipeline

🖼️ Image preprocessing + augmentation for improved accuracy

🧠 Custom CNN-based classifier built using TensorFlow/Keras

📊 Training history visualization (accuracy & loss curves)

📁 Dataset loader with easy directory structure

⚙️ Works on Google Colab, Jupyter Notebook, and VS Code

🔧 Highly scalable for real-time waste classification apps

🛠️ Tech Stack
Component	Technology Used
Language	Python
Frameworks	TensorFlow, Keras
Libraries	NumPy, Pandas, Matplotlib, OpenCV, Scikit-learn
Platform	Google Colab / VS Code
Model Type	Convolutional Neural Network (CNN)
📂 Project Structure
AI_Sustainability_Project_Week1/
│── dataset/
│   ├── TRAIN/
│   ├── TEST/
│── src/
│   ├── train.py
│   ├── preprocessing.py
│   ├── predict.py
│── saved_model/
│── README.md
│── requirements.txt

🔧 Installation
1️⃣ Clone the Repository
git clone https://github.com/sreyeshmandly/AI_Sustainability_Project_Week1.git
cd AI_Sustainability_Project_Week1

2️⃣ Install Dependencies

If using Colab, most libraries already exist.
For VS Code / Local system:

pip install -r requirements.txt


OR install manually:

pip install tensorflow numpy pandas matplotlib opencv-python scikit-learn

🚀 How to Run the Project
▶️ Option 1: Run on Google Colab

Upload the project folder to Drive

Open the notebook or .py files

Run all cells

▶️ Option 2: Train the Model Locally (VS Code)

Run preprocessing:

python src/preprocessing.py


Train the CNN model:

python src/train.py


Run prediction on a single image:

python src/predict.py

📊 Results

Achieved high training & validation accuracy

Training curves clearly show learning efficiency

Model performed well across multiple waste categories

You can visualize results using:

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

🗂️ Dataset

This project uses a Waste Classification Dataset, containing categories like:

🥗 Organic Waste

🧴 Recyclable Waste

Plastic

Glass

Paper

Metal

⚠️ Hazardous Waste

You can replace with your own dataset by maintaining the folder structure.

🌍 Future Enhancements

🔹 Deploy as a Web App using Flask/Streamlit
🔹 Deploy as Android App using TensorFlow Lite
🔹 Integrate with IoT Smart Bins (Raspberry Pi)
🔹 Improve accuracy with Transfer Learning (MobileNet, ResNet)
🔹 Add real-time classification using OpenCV Camera Feed

🤝 Contributing

Contributions are welcome!
Feel free to open an Issue or Pull Request.

⭐ Show Your Support

If you like this project, please give it a star ⭐ on GitHub — it helps a lot!
