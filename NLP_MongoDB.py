'''
NLP python code using NLTK and image processing in reading prescription claimed by a patient, 
convert it to a text and detec fraud cases based on a pre-trained system
ChatGPT
Here's a basic outline of a Python code that demonstrates how you can use NLTK for natural 
language processing and image processing to read prescriptions claimed by a patient, 
convert them to text, and detect fraud cases based on a pre-trained system.

This code assumes that you have installed Tesseract OCR, OpenCV (cv2), and pymongo 
libraries. It first preprocesses the extracted text using NLTK by tokenizing, 
removing stopwords, and converting to lowercase. Then, it detects fraud cases 
based on predefined rules. Finally, it stores the detected fraud cases in a MongoDB database. 
Adjust the fraud detection logic according to your specific requirements.
'''

python
Copy code
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pytesseract
import cv2
import pymongo

# Download NLTK resources
nltk.download("punkt")
nltk.download("stopwords")

# Function to preprocess text
def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalpha()]
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if not word in stop_words]
    return " ".join(tokens)

# Function to extract text from image using Tesseract OCR
def extract_text_from_image(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)
    return text

# Function to detect fraud cases
def detect_fraud_prescriptions(text):
    # Placeholder for fraud detection logic
    if "dangerous drug" in text:
        return True
    else:
        return False

# Read prescription images, extract text, and detect fraud cases
prescription_images = ["prescription1.jpg", "prescription2.jpg", "prescription3.jpg"]
fraud_cases = []

for image_path in prescription_images:
    text = extract_text_from_image(image_path)
    preprocessed_text = preprocess_text(text)
    if detect_fraud_prescriptions(preprocessed_text):
        fraud_cases.append({"image_path": image_path, "text": text})

# Store fraud cases in MongoDB
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["fraud_detection_db"]
collection = db["fraud_cases"]
for case in fraud_cases:
    collection.insert_one(case)

print("Fraud cases detected and stored in MongoDB.")
