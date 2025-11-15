🌱 Waste Classification using CNN
AI-powered waste detection for sustainable smart cities
<div align="center">

🚮♻️ Automating Waste Segregation using Deep Learning
📷 Built with TensorFlow/Keras
⚡ Powered by Convolutional Neural Networks (CNNs)
🌍 Designed for Smart Bins, IoT, and Green AI Solutions

</div>
📌 Project Overview

Waste Classification using CNN is a deep learning project that automatically classifies waste images into categories such as:

Organic Waste

Recyclable Waste (Plastic, Paper, Glass, Metal)

Hazardous Waste

The system helps improve waste segregation, supporting clean cities, smart waste-management systems, and sustainable development initiatives.

This model is built using TensorFlow/Keras, and runs smoothly on Google Colab, Jupyter Notebook, or VS Code.

✨ Key Features

🚀 Complete Deep Learning Pipeline

🖼️ Image Preprocessing + Augmentation

🧠 Custom CNN-based Waste Classifier

📊 Visualization of Accuracy & Loss Curves

📁 Modular Code Structure (train/preprocess/predict)

⚙️ Compatible with VS Code, Colab, and Jupyter

📦 Ready for Deployment (Web / Mobile / IoT)

🛠️ Tech Stack
Component	Technology
Language	Python
Frameworks	TensorFlow, Keras
Libraries	NumPy, Matplotlib, Pandas, OpenCV, Scikit-learn
Platform	Google Colab / VS Code
Model Type	Convolutional Neural Network (CNN)
📂 Project Structure
AI_Sustainability_Project_Week1/
AI_Sustainability_Project_Week1/
│── dataset/
│   ├── TRAIN/
│   └── TEST/
│
│── src/
│   ├── train.py
│   ├── preprocessing.py
│   └── predict.py
│
│── saved_model/
│── requirements.txt
│── README.md


🔧 Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/sreyeshmandly/AI_Sustainability_Project_Week1.git
cd AI_Sustainability_Project_Week1

2️⃣ Install Dependencies

✔️ For Google Colab — most packages already available
✔️ For local setup:

Install using requirements.txt:

pip install -r requirements.txt


OR install manually:

pip install tensorflow numpy pandas matplotlib opencv-python scikit-learn

🚀 How to Run the Project
▶️ Run on Google Colab

Upload the entire project folder to Google Drive

Open .ipynb or .py files in Colab

Run all cells to train and test the model

▶️ Run Locally (VS Code / Jupyter)

📌 Preprocess Dataset

python src/preprocessing.py


📌 Train the CNN Model

python src/train.py


📌 Run Prediction on New Image

python src/predict.py

📊 Results & Performance

Achieved high accuracy on validation data

Smooth convergence during training

Model able to generalize well across multiple waste types

Clear visualization for understanding model performance:

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

🗂️ Dataset Information

This project uses a structured waste dataset with categories such as:

🥗 Organic Waste

Food waste

Leaves

Vegetables

🧴 Recyclable Waste

Plastic

Glass

Metal

Paper

⚠️ Hazardous Waste

Batteries

Chemicals

Medical waste

You can replace the dataset with your own — just maintain the folder structure.

🌍 Future Enhancements

🔹 Deploy as Web App using Streamlit/Flask
🔹 Develop Android App using TensorFlow Lite
🔹 Integrate with IoT Smart Bins (Raspberry Pi)
🔹 Improve accuracy with MobileNet / ResNet Transfer Learning
🔹 Add real-time camera detection using OpenCV

🤝 Contributing

Contributions, improvements, and suggestions are welcome!
Feel free to create an Issue or Pull Request.

⭐ Support the Project

If you found this project helpful, please ⭐ star the repository.
Your support motivates further development!
