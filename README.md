# deeplearning
# 🎙️ Speech Command Classification using CNN & Transfer Learning

This project implements a deep learning pipeline to classify **spoken commands** using **spectrogram images**. It explores both custom Convolutional Neural Networks (CNNs) and transfer learning with pre-trained architectures like **GoogLeNet** and **ResNet50**, implemented in **MATLAB**.

---

## 🚀 Project Highlights

- Converted speech audio into **spectrogram images**
- Built a custom **CNN classifier** with data augmentation
- Tuned multiple **CNN architectures** (filters × layers)
- Implemented **Ensemble Learning** with majority voting
- Fine-tuned **GoogLeNet** and **ResNet50** using transfer learning
- Evaluated models using **confusion matrices**, **accuracy plots**, and **training curves**

---

## 📊 Technologies Used

- **MATLAB** (Deep Learning Toolbox)
- **CNN (custom-built)**
- **Transfer Learning**: GoogLeNet, ResNet50
- **ImageDatastore**, **AugmentedImageDatastore**
- **Data Augmentation** (scaling, translation, flipping, rotation)
- **Confusion Matrices**, Accuracy Plots

---

## 🔍 Key Results

- Models trained using 50 epochs, batch size 32, Adam optimizer.

---

## 📷 Sample Output

![Confusion Matrix](result%20of%20googlenet.png)

---

## 📌 How to Run

1. Open `ACS61011_speech_project.m` in MATLAB.
2. Ensure `speechImageData/` folder is in the same directory.
3. Run the script — it automatically performs data loading, augmentation, model training, and evaluation.

---

## ✍️ Author

**Animesh Sandhu**  
📍 Delhi, India  
📧 Animeshsandhu75@gmail.com  

---

## 📄 License

This project is for educational and demonstration purposes. Contact for reuse or collaboration.
