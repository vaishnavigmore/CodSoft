First.py

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help="Image Path")
args = vars(ap.parse_args())
img_path = args['image']
def extract_features(filename, model):
try:
           image = Image.open(filename)
except:
           print("ERROR: Can't open image! Ensure that image path and extension is correct")
       image = image.resize((299,299))
       image = np.array(image)
      # for 4 channels images, we need to convert them into 3 channels
if image.shape[2] == 4:
           image = image[..., :3]
       image = np.expand_dims(image, axis=0)
       image = image/127.5
       image = image - 1.0
       feature = model.predict(image)
return feature
def word_for_id(integer, tokenizer):
for word, index in tokenizer.word_index.items():
if index == integer:
return word
return None
def generate_desc(model, tokenizer, photo, max_length):
   in_text = 'start'
for i in range(max_length):
       sequence = tokenizer.texts_to_sequences([in_text])[0]
       sequence = pad_sequences([sequence], maxlen=max_length)
       pred = model.predict([photo,sequence], verbose=0)
       pred = np.argmax(pred)
       word = word_for_id(pred, tokenizer)
if word is None:
           break
       in_text += ' ' + word
if word == 'end':
           break
return in_text
max_length = 32
tokenizer = load(open("tokenizer.p","rb"))
model = load_model('models/model_9.h5')
xception_model = Xception(include_top=False, pooling="avg")
photo = extract_features(img_path, xception_model)
img = Image.open(img_path)
description = generate_desc(model, tokenizer, photo, max_length)
print("nn")
print(description)
plt.imshow(img)

Features.py

def extract_features(directory):
       model = Xception( include_top=False, pooling='avg' )
       features = {}
for pic in tqdm(os.listdir(dirc)):
           file = dirc + "/" + pic
           image = Image.open(file)
           image = image.resize((299,299))
           image = np.expand_dims(image, axis=0)
          #image = preprocess_input(image)
           image = image/127.5
           image = image - 1.0
           feature = model.predict(image)
           features[img] = feature
return features
#2048 feature vector
features = extract_features(dataset_images)
dump(features, open("features.p","wb"))
#to directly load the features from the pickle file.
features = load(open("features.p","rb"))

Flick.py

#load the data
def load_photos(filename):
   file = load_doc(filename)
   photos = file.split("n")[:-1]
return photos
def load_clean_descriptions(filename, photos):
  #loading clean_descriptions
   file = load_doc(filename)
   descriptions = {}
for line in file.split("n"):
       words = line.split()
if len(words)<1 :
           continue
       image, image_caption = words[0], words[1:]
if image in photos:
if image not in descriptions:
               descriptions[image] = []
           desc = ' ' + " ".join(image_caption) + ' '
           descriptions[image].append(desc)
return descriptions
def load_features(photos):
  #loading all features
   all_features = load(open("features.p","rb"))
  #selecting only needed features
   features = {k:all_features[k] for k in photos}
return features
filename = dataset_text + "/" + "Flickr_8k.trainImages.txt"
#train = loading_data(filename)
train_imgs = load_photos(filename)
train_descriptions = load_clean_descriptions("descriptions.txt", train_imgs)
train_features = load_features(train_imgs)

Tokenizer.py

#convert dictionary to clear list of descriptions
def dict_to_list(descriptions):
   all_desc = []
for key in descriptions.keys():
       [all_desc.append(d) for d in descriptions[key]]
return all_desc
#creating tokenizer class
#this will vectorise text corpus
#each integer will represent token in dictionary
from keras.preprocessing.text import Tokenizer
def create_tokenizer(descriptions):
   desc_list = dict_to_list(descriptions)
   tokenizer = Tokenizer()
   tokenizer.fit_on_texts(desc_list)
return tokenizer
# give each word an index, and store that into tokenizer.p pickle file
tokenizer = create_tokenizer(train_descriptions)
dump(tokenizer, open('tokenizer.p', 'wb'))
vocab_size = len(tokenizer.word_index) + 1
Vocab_size #The size of our vocabulary is 7577 words.
#calculate maximum length of descriptions to decide the model structure parameters.
def max_length(descriptions):
   desc_list = dict_to_list(descriptions)
return max(len(d.split()) for d in desc_list)
max_length = max_length(descriptions)
Max_length #Max_length of description is 32

Generator.py

#data generator, used by model.fit_generator()
def data_generator(descriptions, features, tokenizer, max_length):
while 1:
for key, description_list in descriptions.items():
          #retrieve photo features
           feature = features[key][0]
           inp_image, inp_seq, op_word = create_sequences(tokenizer, max_length, description_list, feature)
           yield [[inp_image, inp_sequence], op_word]
def create_sequences(tokenizer, max_length, desc_list, feature):
   x_1, x_2, y = list(), list(), list()
  # move through each description for the image
for desc in desc_list:
      # encode the sequence
       seq = tokenizer.texts_to_sequences([desc])[0]
      # divide one sequence into various X,y pairs
for i in range(1, len(seq)):
          # divide into input and output pair
           in_seq, out_seq = seq[:i], seq[i]
          # pad input sequence
           in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
          # encode output sequence
           out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
          # store
           x_1.append(feature)
           x_2.append(in_seq)
           y.append(out_seq)
return np.array(X_1), np.array(X_2), np.array(y)
#To check the shape of the input and output for your model
[a,b],c = next(data_generator(train_descriptions, features, tokenizer, max_length))
a.shape, b.shape, c.shape
#((47, 2048), (47, 32), (47, 7577))

Adding CNN & RNN 

from keras.utils import plot_model
# define the captioning model
def define_model(vocab_size, max_length):
  # features from the CNN model compressed from 2048 to 256 nodes
   inputs1 = Input(shape=(2048,))
   fe1 = Dropout(0.5)(inputs1)
   fe2 = Dense(256, activation='relu')(fe1)
  # LSTM sequence model
   inputs2 = Input(shape=(max_length,))
   se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
   se2 = Dropout(0.5)(se1)
   se3 = LSTM(256)(se2)
  # Merging both models
   decoder1 = add([fe2, se3])
   decoder2 = Dense(256, activation='relu')(decoder1)
   outputs = Dense(vocab_size, activation='softmax')(decoder2)
  # merge it [image, seq] [word]
   model = Model(inputs=[inputs1, inputs2], outputs=outputs)
   model.compile(loss='categorical_crossentropy', optimizer='adam')
  # summarize model
   print(model.summary())
   plot_model(model, to_file='model.png', show_shapes=True)
return model

# train our model
print('Dataset: ', len(train_imgs))
print('Descriptions: train=', len(train_descriptions))
print('Photos: train=', len(train_features))
print('Vocabulary Size:', vocab_size)
print('Description Length: ', max_length)
model = define_model(vocab_size, max_length)
epochs = 10
steps = len(train_descriptions)
# creating a directory named models to save our models
os.mkdir("models")
for i in range(epochs):
   generator = data_generator(train_descriptions, train_features, tokenizer, max_length)
   model.fit_generator(generator, epochs=1, steps_per_epoch= steps, verbose=1)
   model.save("models/model_" + str(i) + ".h5")

test_caption.py

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help="Image Path")
args = vars(ap.parse_args())
img_path = args['image']
def extract_features(filename, model):
try:
           image = Image.open(filename)
except:
           print("ERROR: Can't open image! Ensure that image path and extension is correct")
       image = image.resize((299,299))
       image = np.array(image)
      # for 4 channels images, we need to convert them into 3 channels
if image.shape[2] == 4:
           image = image[..., :3]
       image = np.expand_dims(image, axis=0)
       image = image/127.5
       image = image - 1.0
       feature = model.predict(image)
return feature
def word_for_id(integer, tokenizer):
for word, index in tokenizer.word_index.items():
if index == integer:
return word
return None
def generate_desc(model, tokenizer, photo, max_length):
   in_text = 'start'
for i in range(max_length):
       sequence = tokenizer.texts_to_sequences([in_text])[0]
       sequence = pad_sequences([sequence], maxlen=max_length)
       pred = model.predict([photo,sequence], verbose=0)
       pred = np.argmax(pred)
       word = word_for_id(pred, tokenizer)
if word is None:
           break
       in_text += ' ' + word
if word == 'end':
           break
return in_text
max_length = 32
tokenizer = load(open("tokenizer.p","rb"))
model = load_model('models/model_9.h5')
xception_model = Xception(include_top=False, pooling="avg")
photo = extract_features(img_path, xception_model)
img = Image.open(img_path)
description = generate_desc(model, tokenizer, photo, max_length)
print("nn")
print(description)
plt.imshow(img)