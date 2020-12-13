import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models
import tkinter as tk
from tkinter import filedialog

# Assigning data to tuples

(training_images, training_labels), (testing_images, testing_labels)=datasets.cifar10.load_data()
training_images, testing_images = training_images / 255, testing_images/ 255

class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# To Show Data-set sample
for i in range(16):
    plt.subplot(4,4,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(training_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[training_labels[i][0]])
plt.show()

# To divide Training and Testing data
training_images = training_images[:20000]
training_labels = training_labels[:20000]
testing_images = testing_images[:4000]
testing_labels = testing_labels[:4000]

#load previously trained model
model = models.load_model('image_classifier.model')

# if perviously trained model not found train new one
if model == None:

    #Structure of Neural Network
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(64, (3,3), activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(64, (3,3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(64, activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(training_images, training_labels, epochs=10, validation_data=(testing_images, testing_labels))

    # Validating trained model
    loss, accuracy = model.evaluate(testing_images,testing_labels)
    print(f"Loss: {loss}")
    print(f"Accuracy: {accuracy}")

    # Save Model
    model.save('image_classifier.model')

#Taking Image input using File selection window
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename()
img = cv.imread(file_path)
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

#Predicting/Classifying image  
prediction = model.predict(np.array([img]) / 255)
index = np.argmax(prediction)
plt.xticks([])
plt.yticks([])
plt.imshow(img, cmap=plt.cm.binary)
plt.xlabel(f'I found a {class_names[index]}')
plt.show()
