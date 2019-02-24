import warnings
warnings.filterwarnings("ignore", category = FutureWarning)

import os
import keras
import h5py
import librosa
import itertools
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import BatchNormalization

from mysite.settings import *

from keras.models import model_from_json





def plsWork(docName) :



	uploadedSongPath = MEDIA_ROOT + docName



	def splitsongs(X, y, window = 0.1, overlap = 0.5):
		"""
		@description: Method to split a song into multiple songs using overlapping windows
		"""

		# Empty lists to hold our results
		temp_X = []
		temp_y = []

		# Get the input song array size
		xshape = X.shape[0]
		chunk = int(xshape*window)
		offset = int(chunk*(1.-overlap))

		# Split the song and create new ones on windows
		spsong = [X[i:i+chunk] for i in range(0, xshape - chunk + offset, offset)]
		for s in spsong:
			temp_X.append(s)
			temp_y.append(y)

		return np.array(temp_X), np.array(temp_y)




	def to_melspectrogram(songs, n_fft = 1024, hop_length = 512):
		"""
		@description: Method to convert a list of songs to a np array of melspectrograms
		"""

		# Transformation function
		melspec = lambda x: librosa.feature.melspectrogram(x, n_fft = n_fft,hop_length = hop_length)[:,:,np.newaxis]

		# map transformation of input songs to melspectrogram using log-scale
		tsongs = map(melspec, songs)
		return np.array(list(tsongs))


	

	def read_data(src_dir, genres, song_samples, spec_format, debug = True):    
	    # Empty array of dicts with the processed features from all files
	    arr_specs = []
	    arr_genres = []
	    #print(src_dir)
	    # Read files from the folders
	    for x,_ in genres.items():
	        folder = src_dir + "/" + x
	        #print(folder)
	        for root, subdirs, files in os.walk(folder):
	            for file in files:
	                # Read the audio file
	                file_name = folder + "/" + file
	                signal, sr = librosa.load(file_name)
	                signal = signal[:song_samples]
	                
	                # Debug process
	                if debug:
	                    #print("Reading file: {}".format(file_name))
	                    pass
	                
	                # Convert to dataset of spectograms/melspectograms
	                signals, y = splitsongs(signal, genres[x])
	                
	                # Convert to "spec" representation
	                specs = spec_format(signals)
	                
	                # Save files
	                arr_genres.extend(y)
	                arr_specs.extend(specs)
	                
	                
	    return np.array(arr_specs), np.array(arr_genres)




	#======================================================================


	#In[5]:

	# Parameters
	gtzan_dir = STATIC_URL + 'convWavFiles'  #this is where the directory for where all genre folders are goes.
	song_samples = 660000
	

	genres = {
				'metal': 0, 
				'disco': 1, 
				'classical': 2, 
				'hiphop': 3, 
				'jazz': 4, 
	          	'country': 5, 
	          	'pop': 6, 
	          	'blues': 7, 
	          	'reggae': 8, 
	          	'rock': 9
	          }


	# Read the data
	X, y = read_data(gtzan_dir, genres, song_samples, to_melspectrogram, debug=False)
	# In[40]:
	np.save('x_gtzan_npy.npy', X)
	np.save('y_gtzan_npy.npy', y)
	# In[6]:
	# One hot encoding of the labels
	y = to_categorical(y)
	#print(y)
	# # Dataset Split
	# In[7]
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify = y)
	# In[8]:

	#print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)





	#======================================================================




	



	logic_path = BASE_DIR + '/xlsite/logic/' 




	json_file = open(logic_path+'model.json', 'r') #directory of saved json file of model
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	loaded_model.load_weights(logic_path+"model1.h5") #directory of model .h5 file
	#print("Loaded model from disk")
	 
	# evaluate loaded model on test data
	# loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
	loaded_model.compile(loss=keras.losses.categorical_crossentropy,
	              optimizer=keras.optimizers.Adam(),
	              metrics=['accuracy'])

	score = loaded_model.evaluate(X_test, y_test, verbose=0)
	#print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))


	# In[10]:


	def read_data_input(dir,song_samples, spec_format, debug = True):
	    file_name = dir
	    signal, sr = librosa.load(file_name)
	    signal = signal[:song_samples]
	    arr_specs = []            
	    # Debug process
	    if debug:
	        #print("Reading file: {}".format(file_name))
	        pass
	                
	    # Convert to dataset of spectograms/melspectograms
	    signals = splitsongs_test(signal)

	    # Convert to "spec" representation
	    specs = spec_format(signals)

	    # Save files
	    #arr_genres.extend(y)
	    arr_specs.extend(specs)

	                
	    return np.array(arr_specs)

	def splitsongs_test(X, window = 0.1, overlap = 0.5):
	    # Empty lists to hold our results
	    temp_X = []

	    # Get the input song array size
	    xshape = X.shape[0]
	    chunk = int(xshape*window)
	    offset = int(chunk*(1.-overlap))
	    
	    # Split the song and create new ones on windows
	    spsong = [X[i:i+chunk] for i in range(0, xshape - chunk + offset, offset)]
	    for s in spsong:
	        temp_X.append(s)

	    return np.array(temp_X)


	dir = uploadedSongPath #directory of uploaded song by user


	test_input_file = read_data_input(dir,song_samples, to_melspectrogram, debug = True)
	arrayAnswer = loaded_model.predict_classes(test_input_file)
	

	finalFoundGenre = np.bincount(arrayAnswer).argmax()

	#print(finalFoundGenre)


	if finalFoundGenre == 0 :
	    answer = "Metal"
	elif finalFoundGenre == 1 :
	    answer = "Disco"
	elif finalFoundGenre == 2 :
	    answer = "Classical"
	elif finalFoundGenre == 3 :
	    answer = "HipHop"
	elif finalFoundGenre == 4 :
	    answer = "Jazz"
	elif finalFoundGenre == 5 :
	    answer = "Country"
	elif finalFoundGenre == 6 :
	    answer = "Pop"
	elif finalFoundGenre == 7 :
	    answer = "Blues"
	elif finalFoundGenre == 8 :
	    answer = "Reggae"
	else :
	    answer = "Rock"


	return("Uploaded file's genre is : " + answer)







