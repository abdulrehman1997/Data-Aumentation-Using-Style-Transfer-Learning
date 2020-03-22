from keras.models import Model, load_model
from keras.layers import Dense
from keras.applications.vgg16 import VGG16
import os
import numpy as np
from PIL import Image
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.optimizers import SGD

model = VGG16(include_top=True, input_shape=(224, 224, 3))
layers = [layer for layer in model.layers]
for i in range(0, 19):
	layers[i].trainable = False
# layers[20].trainable = False
x = Dense(2,activation= 'softmax',name='output')(layers[22].output)
model = Model(input=model.input, output=x)
# model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# model.save('test.h5')

# exit(0)
# model = load_model('test.h5')

path = '/Users/abdulrehman/Desktop/Project/data'

classes=['male','female']

x=[]
y=[]
for fol in classes:
    imgfiles=[file for file in os.listdir(path+'/'+fol) if file.endswith(".jpg")]
    for img in imgfiles:
        im=Image.open(path+'/'+fol+'/'+img)
        im=np.asarray(im, dtype="float32")
        x.append(im)
        y.append(fol)

x=np.array(x)
y=np.array(y)

print(x.shape)
print(y.shape)

batch_size=32
nb_classes=len(classes)
nb_epoch=10
learning_rate = 0.1
decay_rate = learning_rate / nb_epoch
momentum = 0.8
sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
model.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])

x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=0.2,random_state=4)

uniques, id_train=np.unique(y_train,return_inverse=True)
Y_train=np_utils.to_categorical(id_train,nb_classes)
uniques, id_test=np.unique(y_test,return_inverse=True)
Y_test=np_utils.to_categorical(id_test,nb_classes)

model.fit(x_train,Y_train,batch_size=batch_size,epochs=nb_epoch,verbose=1,validation_data=(x_test, Y_test))

scores = model.evaluate(x_test, Y_test, verbose=0)
print("loss ",scores[0],"accuracy",scores[1])