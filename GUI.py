# -*- coding: utf-8 -*-

import cv2
import numpy as np
import tkinter as tk
from PIL import ImageTk, Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import img_to_array
import pyttsx3


# Load the trained model
model = load_model(r"enter trained model path")

# Initialize the text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)

# Create the GUI window
window = tk.Tk()
window.title("Currency Determination")

# Create a label to display the output
output_label = tk.Label(window, text="", font=("Helvetica", 50))
output_label.pack()

# Create a label for the additional image
additional_image_label = tk.Label(window)
additional_image_label.pack()

# Dictionary mapping predicted class to image path
image_paths = {
    "100 rupees": r" path ",
    "200 rupees": r"path ",
    "500 rupees": r"path ",
    "50 rupees": r"path ",
    "10 rupees": r"path ",
    "20 rupees": r"path "
}

# Function to classify the captured image
def classify_image():
    # Capture an image from the camera
    camera = cv2.VideoCapture(0)
    return_value, img = camera.read()
    cv2.imwrite('test.png', img)
    del(camera)

    # Load and preprocess the captured image
    inp = image.load_img(r" enter path of captured image", target_size=(224, 224))
    test_image = img_to_array(inp)
    test_image = np.expand_dims(test_image, axis=0)

    # Make a prediction using the model
    predictions = model.predict(test_image)
    predicted_class = np.argmax(predictions, axis=1)[0]
    
    # Convert the predicted class to text
    classes = ["100 rupees", "200 rupees", "500 rupees", "50 rupees", "10 rupees", "20 rupees","white paper"]
    predicted_text = classes[predicted_class]
    
    # Update the output label
    output_label.config(text=predicted_text)

    # Convert the predicted text to speech
    engine.say(predicted_text)
    engine.runAndWait()
    
    # Load the additional image based on predicted class
    additional_image_path = image_paths.get(predicted_text)
    additional_image = Image.open(additional_image_path)
    #additional_image = additional_image.resize((400, 400), Image.ANTIALIAS)
    additional_image_tk = ImageTk.PhotoImage(additional_image)
    
    # Update the additional image label
    additional_image_label.config(image=additional_image_tk)
    additional_image_label.image = additional_image_tk  # Store a reference to avoid garbage collection

engine.say("Please press Enter to capture an image.")
engine.runAndWait()

def on_key_press(event):
    if event.keysym == 'Return':
        classify_image()
    elif event.keysym == 'Escape':
        exit_application()

def exit_application():
    window.destroy()

window.bind('<Key>', on_key_press)
window.mainloop()
