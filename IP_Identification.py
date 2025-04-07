import cv2
import numpy as np
import os
from skimage.feature import local_binary_pattern
from numpy.fft import fft2, fftshift
from sklearn.model_selection import train_test_split
import pywt


def save_templates(templates, file_path):
    np.save(file_path, templates)

def classify_image(image, templates):
    features = extract_advanced_features(image)
    min_distance = float('inf')
    predicted_label = None
    for label, template_features in templates.items():
        distance = np.linalg.norm(features - template_features)
        if distance < min_distance:
            min_distance = distance
            predicted_label = label
    return predicted_label



def extract_dwt_features(image):
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Perform 2D Discrete Wavelet Transform
    coeffs2 = pywt.dwt2(gray_image, 'haar')
    LL, (LH, HL, HH) = coeffs2
    # Flatten and combine all the DWT coefficients into a single feature vector
    dwt_features = np.hstack([c.flatten() for c in [LL, LH, HL, HH]])
    return dwt_features

def extract_advanced_features(image):
    if image is None:
        return None
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    lbp = local_binary_pattern(gray_image, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 27), density=True)
    
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    color_hist = cv2.calcHist([hsv_image], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    color_hist = cv2.normalize(color_hist, color_hist).flatten()

    dct = cv2.dct(np.float32(gray_image) / 255.0)
    dct = dct[:20, :20].flatten()

    dft = fft2(gray_image)
    dft_shift = fftshift(dft)
    magnitude_spectrum = 20 * np.log(np.abs(dft_shift))
    dft_features = magnitude_spectrum[:20, :20].flatten()
    dwt_features = extract_dwt_features(image)

    features = np.hstack((lbp_hist, color_hist, dct, dft_features,dwt_features))
    return features

def load_images_and_labels_from_folder(folder):
    data = []
    labels = []
    for label in os.listdir(folder):
        class_path = os.path.join(folder, label)
        for filename in os.listdir(class_path):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")) and filename != "Thumbs.db":
                img_path = os.path.join(class_path, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, (128, 128))
                    data.append(img)
                    labels.append(label)
    return data, labels

def create_templates(data, labels):
    unique_labels = set(labels)
    templates = {}
    for label in unique_labels:
        label_data = [data[i] for i in range(len(data)) if labels[i] == label]
        features_list = [extract_advanced_features(img) for img in label_data if img is not None]
        templates[label] = np.mean(features_list, axis=0)
    return templates

def calculate_accuracy(data, labels, templates):
    correct_predictions = 0
    for img, true_label in zip(data, labels):
        predicted_label = classify_image(img, templates)
        if predicted_label == true_label:
            correct_predictions += 1
    return correct_predictions / len(data)

def load_templates(file_path):
    return np.load(file_path, allow_pickle=True).item()

def classify_input_image(image_path, templates):
    image = cv2.imread(image_path)
    if image is None:
        print("Image not found or unable to load.")
        return
    image = cv2.resize(image, (128, 128))  # Resize to match training data
    predicted_label = classify_image(image, templates)
    return predicted_label



def main():
    dataset_path = 'dataset'  # Actual path to your dataset
    data, labels = load_images_and_labels_from_folder(dataset_path)
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)

    templates_path = 'templates.npy'
    if os.path.exists(templates_path):
        templates = load_templates(templates_path)
    else:
        # Create templates if not already saved
        templates = create_templates(train_data, train_labels)
        save_templates(templates, templates_path)

    accuracy = calculate_accuracy(test_data, test_labels, templates)
    print(f"Classification accuracy: {accuracy:.2%}")


    # Add input image classification functionality
    # You can replace 'image_1.jpg' with any image file you want to classify
    input_image_path = 'dataset\Eczema\\040106HB.jpg'  
    prediction = classify_input_image(input_image_path, templates)
    if prediction is not None:
        print(f"The predicted label for the input image is: {prediction}")
    else:
        print("Unable to make a prediction.")


def classify_input_image_from_web(image, templates):
    if image is None:
        print("Image not found or unable to load.")
        return "Unable to load image"
    predicted_label = classify_image(image, templates)
    return predicted_label

# Call the main function
if __name__ == "__main__":
    main()