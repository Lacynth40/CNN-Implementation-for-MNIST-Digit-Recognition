Drake Farrokhi: A00536349
Cynthia Curtis:  1324152
Dan Roberts: A00067239

CNN Implementation for MNIST Digit Recognition

For this project, we started out implementing our groups normal approach to these projects, which is a brute-force coding method going on while the rest of the group finds code snippets that would make this easier. That method didn’t survive long, as our brute-force coder pointed out that the dataset we were given to work on this project with did not match the normal MSINT dataset. So, we cautiously began with data preprocessing to locate all possible differences. 
from ucimlrepo import fetch_ucirepo   
# fetch dataset 
optical_recognition_of_handwritten_digits = fetch_ucirepo(id=80)   
# data (as pandas dataframes) 
X = optical_recognition_of_handwritten_digits.data.features 
y = optical_recognition_of_handwritten_digits.data.targets
# metadata 
print(optical_recognition_of_handwritten_digits.metadata)
{'uci_id': 80, 'name': 'Optical Recognition of Handwritten Digits', 'repository_url': 'https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits', 'data_url': 'https://archive.ics.uci.edu/static/public/80/data.csv', 'abstract': 'Two versions of this database available; see folder', 'area': 'Computer Science', 'tasks': ['Classification'], 'characteristics': ['Multivariate'], 'num_instances': 5620, 'num_features': 64, 'feature_types': ['Integer'], 'demographics': [], 'target_col': ['class'], 'index_col': None, 'has_missing_values': 'no', 'missing_values_symbol': None, 'year_of_dataset_creation': 1998, 'last_updated': 'Wed Aug 23 2023', 'dataset_doi': '10.24432/C50P49', 'creators': ['E. Alpaydin', 'C. Kaynak'], 'intro_paper': {'title': 'Methods of Combining Multiple Classifiers and Their Applications to Handwritten Digit Recognition', 'authors': 'C. Kaynak', 'published_in': 'MSc Thesis, Institute of Graduate Studies in Science and Engineering, Bogazici University', 'year': 1995, 'url': None, 'doi': None}, 'additional_info': {'summary': 'We used preprocessing programs made available by NIST to extract normalized bitmaps of handwritten digits from a preprinted form. From a total of 43 people, 30 contributed to the training set and different 13 to the test set. 32x32 bitmaps are divided into nonoverlapping blocks of 4x4 and the number of on pixels are counted in each block. This generates an input matrix of 8x8 where each element is an integer in the range 0..16. This reduces dimensionality and gives invariance to small distortions.\r\n\r\nFor info on NIST preprocessing routines, see M. D. Garris, J. L. Blue, G. T. Candela, D. L. Dimmick, J. Geist, P. J. Grother, S. A. Janet, and C. L. Wilson, NIST Form-Based Handprint Recognition System, NISTIR 5469, 1994.', 'purpose': None, 'funded_by': None, 'instances_represent': None, 'recommended_data_splits': None, 'sensitive_data': None, 'preprocessing_description': None, 'variable_info': 'All input attributes are integers in the range 0..16.\r\nThe last attribute is the class code 0..9', 'citation': None}}
# variable information 
print(optical_recognition_of_handwritten_digits.variables)
           name     role         type demographic description units  \
0    Attribute1  Feature      Integer        None        None  None   
1    Attribute2  Feature      Integer        None        None  None   
2    Attribute3  Feature      Integer        None        None  None   
3    Attribute4  Feature      Integer        None        None  None   
4    Attribute5  Feature      Integer        None        None  None   
..          ...      ...          ...         ...         ...   ...   
60  Attribute61  Feature      Integer        None        None  None   
61  Attribute62  Feature      Integer        None        None  None   
62  Attribute63  Feature      Integer        None        None  None   
63  Attribute64  Feature      Integer        None        None  None   
64        class   Target  Categorical        None        None  None   

   missing_values  
0              no  
1              no  
2              no  
3              no  
4              no  
..            ...  
60             no  
61             no  
62             no  
63             no  
64             no  

[65 rows x 7 columns]

As you can see in the metadata block, this dataset has already been subject to some forms of data preprocessing. We are getting 8x8 feature arrays for our inputs instead of the standard MSINT inputs of 28x28 images. This will, of course, limit the amount of convolutions we can do on our data. 
	The next step involves building the Convolutional Neural Network (hereafter referred to as CNN), and assigning our activation kernel. Of course, our kernel was already chosen to be ReLu for us, so we didn’t have to do much there. We decided that since we weren’t working with the standard MSINT dataset, we would go with the Keras Sequential model. This came out to only need two layers, as you can’t really divvy up an 8x8 feature array too far. 
  
#Build a CNN from scratch
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

X = X.values.reshape(-1,8,8,1)
y = y.values
#Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# Modify the model architecture by adjusting the pooling parameters
model = keras.Sequential(
    [
        keras.Input(shape=(8, 8, 1)),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(1, 1)),  # Adjust the pooling parameters
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(1, 1)),  # Adjust the pooling parameters
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax"),
    ]
)

Our pooling of course happens within the convolutional code block, so they’re both right there. Our flattening and softmax also happen at this codeblock. It’s just easier that way, and looks better. 
The next step was to compile the model with a few chosen options. Since we needed to document loss and accuracy, we included that in the model compiling. Once that was complete, we went on to training, which ran for ten epochs with a standard 20% validation split. As you can see below, the model increased accuracy and decreased loss as it completed more epochs. 

#Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
#Train the model
history = model.fit(X_train, y_train, epochs=10, validation_split=0.2)

Epoch 1/10
[1m95/95[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 6ms/step - accuracy: 0.5849 - loss: 1.3108 - val_accuracy: 0.9708 - val_loss: 0.1258
Epoch 2/10
[1m95/95[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - accuracy: 0.9334 - loss: 0.2171 - val_accuracy: 0.9788 - val_loss: 0.0833
Epoch 3/10
[1m95/95[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 0.9505 - loss: 0.1670 - val_accuracy: 0.9827 - val_loss: 0.0504
Epoch 4/10
[1m95/95[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 5ms/step - accuracy: 0.9778 - loss: 0.0782 - val_accuracy: 0.9814 - val_loss: 0.0578
Epoch 5/10
[1m95/95[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - accuracy: 0.9774 - loss: 0.0652 - val_accuracy: 0.9907 - val_loss: 0.0307
Epoch 6/10
[1m95/95[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 0.9832 - loss: 0.0511 - val_accuracy: 0.9920 - val_loss: 0.0303
Epoch 7/10
[1m95/95[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 0.9899 - loss: 0.0409 - val_accuracy: 0.9934 - val_loss: 0.0245
Epoch 8/10
[1m95/95[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 6ms/step - accuracy: 0.9898 - loss: 0.0307 - val_accuracy: 0.9947 - val_loss: 0.0196
Epoch 9/10
[1m95/95[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 0.9906 - loss: 0.0353 - val_accuracy: 0.9907 - val_loss: 0.0236
Epoch 10/10
[1m95/95[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 0.9821 - loss: 0.0495 - val_accuracy: 0.9907 - val_loss: 0.0221
#Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)
58/58 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9832 - loss: 0.0444
Test accuracy: 0.9838274717330933

	Now obviously, we aren’t going to just leave it as a bunch of numbers. We have graphs. The first graph is the accuracy and validation accuracy across the ten epochs, and the second graph is the loss and validation loss across the epochs. 
  
#Plot the accuracy
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()
 
#Plot the loss
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim([0, 1])
plt.legend(loc='lower right')  
plt.show()
 

	As you can easily see by these charts, our accuracy steadily climbed, while our loss decreased across the epochs. Obviously, both accuracy and loss are going to become asymptotic, as there is no such thing as a perfect block of code with perfect accuracy and zero loss. But, we got close. 
	The final thing we did was we decided to upload it all to Drake’s GitHub so we would have access to it for a while. We also decided we aren’t going to be using anything involved with an extracurricular club, as that could greatly endanger that club’s funding to use the club’s resources in a class environment. The GitHub repository that our code resides in is right here. 
	In conclusion, we did the best we could with a preprocessed dataset that deviated from the standard MSINT dataset. We used a Keras Sequential model, and had two convolution layers with two maxpool layers. Our accuracy ended up above 99%, and our loss was down to 2%, after ten epochs. I am positive we would have needed three or four convolution layers if we had started with the 28x28 MSINT dataset, instead of the processed 8x8 dataset we were given. As such, we did quite well and produced a model that worked wonderfully. 
