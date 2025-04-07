
import cv2
import numpy as np
import os
from scipy.linalg import svd
from skimage.feature import local_binary_pattern
from pywt import wavedec2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import pickle


def save_model(svm_model, filename):
    svm_model.save(filename)


def load_images_and_labels(folder):
    images = []
    labels = []
    for label in os.listdir(folder):
        path = os.path.join(folder, label)
        for file in os.listdir(path):
            img_path = os.path.join(path, file)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is not None:
                img = cv2.resize(img, (128, 128))
                images.append(img)
                labels.append(label)
    return images, labels

def preprocess_image(image):
    img_normalized = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return img_normalized

def extract_color_histogram(image):
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    return cv2.normalize(hist, hist).flatten()

def extract_lbp_features(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(image_gray, P=24, R=3, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 26), density=True)
    return lbp_hist

def extract_dct_features(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dct = cv2.dct(np.float32(image_gray) / 255.0)
    return dct[:20, :20].flatten()

def extract_dwt_features(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    coeffs = wavedec2(image_gray, 'db4', level=1)
    cA, (cH, cV, cD) = coeffs
    return cA.flatten()

def extract_svd_features(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    U, s, V = svd(image_gray, full_matrices=False)
    return s[:15]

def extract_features(images):
    features = []
    for img in images:
        img = preprocess_image(img)
        features.append(np.hstack([
            extract_color_histogram(img),
            extract_lbp_features(img),
            extract_dct_features(img),
            extract_dwt_features(img),
            extract_svd_features(img)
        ]))
    return features

def train_and_evaluate_svm(features, labels):
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)

    X_train, X_test, y_train, y_test = train_test_split(features, labels_encoded, test_size=0.2, random_state=42)

    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setC(1.0)
    svm.setGamma(0.5)
    svm.train(np.array(X_train, dtype=np.float32), cv2.ml.ROW_SAMPLE, np.array(y_train, dtype=np.int32))

    _, y_pred = svm.predict(np.array(X_test, dtype=np.float32))
    accuracy = accuracy_score(y_test, y_pred.ravel())
    print(f"Classification accuracy: {accuracy:.2%}")

    return svm, le  # Return the trained SVM and the label encoder

def predict_image(svm, le, image):
    img = preprocess_image(image)
    features = np.hstack([
        extract_color_histogram(img),
        extract_lbp_features(img),
        extract_dct_features(img),
        extract_dwt_features(img),
        extract_svd_features(img)
    ])
    _, result = svm.predict(np.array([features], dtype=np.float32))
    predicted_label = le.inverse_transform(result.astype(int))
    return predicted_label[0]

# Main Execution
folder = 'dataset'
images, labels = load_images_and_labels(folder)
features = extract_features(images)
svm_model, label_encoder = train_and_evaluate_svm(features, labels)


svm_filename = 'svm_model.xml'
save_model(svm_model, svm_filename)
print("SVM model saved to", svm_filename)

def load_model(filename):
    svm = cv2.ml.SVM_create()
    svm.load(filename)
    return svm

def save_label_encoder(label_encoder, filename):
    with open(filename, 'wb') as file:
        pickle.dump(label_encoder, file)

def load_label_encoder(filename):
    with open(filename, 'rb') as file:
        label_encoder = pickle.load(file)
    return label_encoder

label_encoder_filename = 'label_encoder.pkl'
save_label_encoder(label_encoder, label_encoder_filename)
print("Label encoder saved to", label_encoder_filename)

# Load the SVM model and label encoder
svm_model_loaded = load_model(svm_filename)
label_encoder_loaded = load_label_encoder(label_encoder_filename)
print("Model and label encoder loaded.")

# Example prediction usage
new_image_path = 'dataset\Eczema\\040072HB.jpg'

def Prediction(new_image_path):
    new_image = cv2.imread(new_image_path, cv2.IMREAD_COLOR)
    if new_image is not None:
        new_image = cv2.resize(new_image, (128, 128))
        prediction = predict_image(svm_model, label_encoder, new_image)
        print(f"The predicted label for the new image is: {prediction}")
    else:
        print("Failed to load the image for prediction.")

def predict_image_path(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is not None:
        image = cv2.resize(image, (128, 128))
        prediction = predict_image(svm_model, label_encoder, image)
        print(f"The predicted label for the new image is: {prediction}")
        return prediction
    else:
        print("Failed to load the image for prediction.")
        return None
