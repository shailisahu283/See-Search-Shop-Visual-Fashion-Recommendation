# **👗 See,Search,Shop:Visual Fashion Recommendation**  

A deep learning-based fashion recommendation system combining **Convolutional Neural Networks (CNNs)** and **Transformers** for efficient image retrieval. This system analyzes query images and retrieves the most similar fashion products from a dataset of images.  

---

## **📑 Table of Contents**  
1. [✨ Features](#features)
2. [🖥️ Tech Stack](#Tech-Stack)
3. [📂 Dataset](#Dataset)
4. [🌟 System Architecture](#system-architecture)  
5. [🚀 Project Workflow](#Project-Workflow)  
6. [📸 Example Outputs](#example-outputs)  
7. [⚙️ Setup Instructions](#setup-instructions)  
8. [▶️ Usage](#usage)  
9. [🔗 Project Highlights](#project-highlights)  
10. [🤝 Contributing](#contributing)
11. [🏆 Acknowledgements](#Acknowledgements)
12. [📜 License](#license)  
13. [👤 Author](#author)  

---

## **✨ Features**  
- **📷 Image-Based Search:** Upload a query/input image to find visually similar fashion products.  
- **🧠 Hybrid Model Architecture:** Combines pre-trained ResNet50 (CNN) for feature extraction and Transformers for advanced sequence modeling.  
- **⚡ Fast and Accurate:** Efficiently handles a large dataset with over 1K images and provides precise recommendations.  
- **📊 Visual Feedback:** Displays query image, top similar images, and their similarity scores.  
- **🛍️ Product Information:** Provides details about recommended products.  

---

## 🖥️ **Tech Stack**  

- **Programming Languages:** Python (PyTorch, TensorFlow)  
- **Libraries:** torchvision, transformers, albumentations, matplotlib  
- **Frameworks:** PyTorch, TensorFlow  
- **Tools:** Google Colab, Kaggle Dataset API  

---

## 📂 **Dataset**  

The project utilizes the [Fashion Product Images Small Dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small) from Kaggle. It contains images of fashion products, which were preprocessed and resized to suit the model's requirements.  

---

## **📂 System Architecture**  
1. **Input:**  
   - Query image provided by the user (e.g., a fashion product).  
2. **Processing:**  
   - **Feature Extraction:** ResNet50 backbone extracts visual features from images.  
   - **Transformer Encoding:** Captures relationships and dependencies among features.  
   - **Similarity Matching:** Computes cosine similarity between query features and dataset features.  
3. **Output:**  
   - Top 10 similar images and their similarity scores.  
   - Detailed product information.  

---

## 🚀 **Project Workflow**  

1. **Dataset Preparation:**  
   - Downloaded 20K images from Kaggle and limited the dataset to the first 1,000 images for experimentation.  
   - Applied augmentations (e.g., random horizontal flips, color jitter) to enhance generalization.  

2. **Model Architecture:**  
   - **ResNet50 Backbone:** Extracts meaningful image features.  
   - **Transformer Encoder:** Processes and refines extracted features for contextual understanding.  
   - **Output Layer:** Produces embeddings used for similarity computations.  

3. **Training Strategy:**  
   - Used contrastive loss to train the model, ensuring features from similar images are close in the embedding space.  

4. **Image Retrieval:**  
   - For a given query image, the system identifies the top-k most similar images using cosine similarity on embeddings.  

---

## 📸 **Visualization**  

Here’s a visual representation of the workflow:  
![image](https://github.com/user-attachments/assets/a8d4bff2-539b-41a5-bdd1-6586b0c50fd7)

---

## **🔗 Project Links**  
- **🖼️ fashion recommendation system:** [Colab Notebook](https://colab.research.google.com/drive/1cjFv3ndA0Cgn2GYD90CF_GwnMzwn1S_b?usp=sharing)

---


## **📸 Example Outputs**  

### **Input Query Image:**  
![image](https://github.com/user-attachments/assets/c573c0e9-22fc-4d87-af47-1afc3c161d8a)
  

---

### **Top Similar Images:**  
![image](https://github.com/user-attachments/assets/898935da-ff65-47dd-a2bb-2d7e689a5d86)

---

### **Product Information Example:**  
![Screenshot 2024-12-02 091626](https://github.com/user-attachments/assets/d418881d-6831-4937-84e1-6549b68d7e5f)


---


## **⚙️ Setup Instructions**  

### **Prerequisites**  
- Python 3.x  
- Libraries: `tensorflow`, `torch`, `transformers`, `opencv-python`, `matplotlib`, and `kagglehub`.  

### **Installation**  
1. Clone the repository:  
   ```bash
   git clone https://github.com/shailisahu283/See-Search-Shop-Visual-Fashion-Recommendation.git
   ```  
2. Install required packages:  
   ```bash
   pip install kagglehub tensorflow transformers torch torchvision albumentations matplotlib
   ```  
3. Download the dataset from Kaggle:  
   ```python
   path = kagglehub.dataset_download("paramaggarwal/fashion-product-images-small")
   ```  
4. Prepare the dataset with the first 1K images:  
   ```python
   # Code for dataset preparation included in the main script.
   ```  

---

## **▶️ Usage**  
1. Run the script in your preferred Python environment (e.g., Google Colab, Jupyter Notebook, or terminal).  
2. Provide a query image and retrieve similar product recommendations.  
3. Use the `display_similar_images` function to visualize results.  

---

## **🔗 Project Highlights**  
- **ResNet50 for CNN Feature Extraction:** Efficiently captures intricate visual features from images.  
- **Transformer Encoder:** Enhances image retrieval performance with advanced sequence modeling.  
- **Contrastive Loss:** Optimized for learning meaningful embeddings.  
- **Interactive Visualization:** Showcases query results with a clean and intuitive UI.  

---

## **🤝 Contributing**  
Contributions are welcome! Here's how you can help:  
1. Fork this repository.  
2. Create a feature branch: `git checkout -b feature-branch-name`.  
3. Commit your changes: `git commit -m 'Add new feature'`.  
4. Push to the branch: `git push origin feature-branch-name`.  
5. Open a pull request for review.  

---

## 🏆 **Acknowledgements**  

- Dataset: [Kaggle](https://www.kaggle.com/)  
- Libraries: PyTorch, torchvision, transformers  

---

## **📜 License**  
This project is licensed under the MIT License.  

---

## **👤 Author**  
**Shaili Sahu**  
📧 Reach out at [shailisahu283@gmail.com](mailto:shailisahu283@gmail.com) for questions or feedback!  

---
