# Currency-Determination-Using-Deep-Learning-
This Currency Determination project aims to develop a deep learning model capable of accurately identifying and determining the denomination of various currencies from images. This project utilizes convolutional neural networks (CNNs), a class of deep learning algorithms well-suited for image classification tasks.This project includes two main components - the deep learning model for currency recognition and a graphical user interface (GUI) implemented in Python for user interaction.


Steps for Currency recognition model:

Data Collection: The project will begin with the collection of a dataset comprising images of various currencies in India. The dataset will include images of  paper currency captured under different lighting conditions and angles to ensure robustness.

Data Preprocessing: Before training the deep learning model, the collected dataset will undergo preprocessing steps such as resizing, normalization, and augmentation to enhance model generalization and performance.

Model Architecture: The core of the project involves designing and implementing a CNN architecture tailored for currency classification. The architecture will consist of convolutional layers followed by pooling layers to extract relevant features from input images. Additionally, fully connected layers and activation function(relu) will be utilized for final classification.

Training: The prepared dataset will be split into training, validation, and testing sets. The deep learning model will be trained using the training set while monitoring performance on the validation set to prevent overfitting. Techniques such as dropout and regularization may be employed to improve model generalization.

Evaluation: The trained model will be evaluated using the testing set to assess its accuracy, precision,  and other relevant metrics. Fine-tuning and optimization may be performed based on evaluation results to enhance model performance.

Deployment: Once the model achieves satisfactory performance, it will be deployed for practical use. This may involve integrating the model into a mobile application or web service or implementing on hardware allowing users to easily determine the denomination of currencies by capturing images using their devices.

GUI implementation in python:

Developed a user-friendly GUI using Python library Tkinter

It uses the device's camera for capturing the  real-time currency  image and  classify using trained deep learning model which is integrated into GUI and it displays recognized currency along with its denomination

