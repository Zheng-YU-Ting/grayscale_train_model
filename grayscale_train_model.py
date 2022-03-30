import numpy as np
import os
from sklearn.metrics import confusion_matrix
import seaborn as sn; sn.set(font_scale=1.4)
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l2, l1
#%%
class_names = ["angry","fear","happy","neutral","sad","surprise"]
#class_names = ["1","2","3"]
class_names_label = {class_name:i for i, class_name in enumerate(class_names)}

IMAGE_SIZE = (128, 128)
#%%
def load_data():
    #datasets = ['trainFER_data-augmentation-Keras', 'testFER_data-augmentation-Keras']
    datasets = ['trainFER', 'testFER']#資料夾
    output = []
    
    # Iterate through training and test sets
    for dataset in datasets:
        images = []
        labels = []
        
        print("Loading {}".format(dataset))
        
        # Iterate through each folder corresponding to a category
        for folder in os.listdir(dataset):
            label = class_names_label[folder]
            
            # Iterate through each image in our folder
            for file in tqdm(os.listdir(os.path.join(dataset, folder))):
                # Get the path name of the image
                img_path = os.path.join(os.path.join(dataset, folder), file)
                #print(img_path)
                # Open and resize the img
                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                #print(image.shape)
                #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                #cv讀照片，顏色莫認為BGR，需轉為RGB
 
                image = cv2.resize(image, IMAGE_SIZE) #改圖大小
                
                # Append the image and its corresponding label to the output
                images.append(image)
                labels.append(label)

        images = np.array(images, dtype = 'float32')
        labels = np.array(labels, dtype = 'int32')   
        
        output.append((images, labels))

    return output
#%%
(train_images, train_labels), (test_images, test_labels) = load_data()

train_images = np.expand_dims(train_images, axis=3)
test_images = np.expand_dims(test_images, axis=3)
#%%
'隨機性'
train_images, train_labels = shuffle(train_images, train_labels, random_state=25)
'標準化'
train_images = train_images / 255.0 
test_images = test_images / 255.0


#%%
'建模'
input_shape = (128, 128, 1)

model = Sequential()
            #特徵擷取
            #濾波器數=特徵圖數 (3*3)                    補0             激活函數relu       每strides格擷取一次特徵
            

model.add(Conv2D(128, (3, 3), input_shape=input_shape, padding='same',activation='relu', strides=1))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))#池化(2*2)降低複雜度
model.add(BatchNormalization())

model.add(Conv2D(64, (3, 3), input_shape=input_shape, padding='same',activation='relu', strides=1))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))#池化(2*2)降低複雜度
model.add(BatchNormalization())

model.add(Conv2D(32, (3, 3), input_shape=input_shape, padding='same',activation='relu', strides=1))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))#池化(2*2)降低複雜度
model.add(BatchNormalization())

model.add(Conv2D(16, (3, 3), input_shape=input_shape, padding='same',activation='relu', strides=1))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))#池化(2*2)降低複雜度
model.add(BatchNormalization())

model.add(Conv2D(8, (3, 3), input_shape=input_shape, padding='same',activation='relu', strides=1))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))#池化(2*2)降低複雜度
model.add(BatchNormalization())

model.add(Conv2D(4, (3, 3), input_shape=input_shape, padding='same',activation='relu', strides=1))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))#池化(2*2)降低複雜度
model.add(BatchNormalization())

#model.add(BatchNormalization(axis=1))
#model.add(BatchNormalization())
#model.add(Dropout(0.1))

model.add(Flatten())

model.add(Dense(512,kernel_regularizer=l2(0.01),activation='relu'))
#model.add(Dense(512,activation='relu'))

model.add(Dense(6, activation='softmax')) #輸出層，分類用softmax

#print(model.summary())
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
#%%
history = model.fit(train_images, train_labels, 
                    validation_data=(test_images, test_labels),
                    #verbose=2,callbacks=[earlyStop],
                    batch_size=64, epochs=100, shuffle=True)
#%%
'模型概況'
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
print('accuracy: ',history.history['accuracy'])
print('val_accuracy: ',history.history['val_accuracy'])

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
print('loss: ',history.history['loss'])
print('val_loss: ',history.history['val_loss'])
#%%
'預測'
predictions = model.predict(test_images)     # Vector of probabilities
#print(predictions)
pred_labels = np.argmax(predictions, axis = 1) # We take the highest probability
#print(pred_labels)

#%%
'混淆矩陣'
CM = confusion_matrix(test_labels, pred_labels)
def accuracy(confusion_matrix):
    diagonal_sum = confusion_matrix.trace()
    sum_of_all_elements = confusion_matrix.sum()
    return diagonal_sum / sum_of_all_elements 

'混淆矩陣視覺化，看錯誤'
ax = plt.axes()
sn.heatmap(CM, annot=True, 
           annot_kws={"size": 10}, 
           xticklabels=class_names, 
           yticklabels=class_names, ax = ax)
ax.set_title('Confusion matrix')
plt.show()
#%%
'存模型'
model.save("FER_64-model")
#model = load_model('CNN_model')

'''1500*128記憶體夠大 | 224*224的460張
input_shape = (64, 64, 1)#彩圖3

model = Sequential()
            #特徵擷取
            #濾波器數=特徵圖數 (3*3)                    補0             激活函數relu       每strides格擷取一次特徵
model.add(Conv2D(64, (3, 3), input_shape=input_shape, padding='same',activation='relu'))
model.add(BatchNormalization(axis=1))
model.add(Conv2D(64, (3, 3), input_shape=input_shape, padding='same',activation='relu'))
model.add(BatchNormalization(axis=1))
model.add(MaxPooling2D(pool_size=(2, 2)))#池化(2*2)降低複雜度
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), input_shape=input_shape, padding='same',activation='relu'))
model.add(BatchNormalization(axis=1))
model.add(Conv2D(128, (3, 3), input_shape=input_shape, padding='same',activation='relu'))
model.add(BatchNormalization(axis=1))
model.add(MaxPooling2D(pool_size=(2, 2)))#池化(2*2)降低複雜度
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(1024,activation='relu'))
model.add(BatchNormalization(axis=1))
model.add(Dropout(0.5))

model.add(Dense(6, activation='softmax')) #輸出層，分類用softmax


print(model.summary())
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
'''