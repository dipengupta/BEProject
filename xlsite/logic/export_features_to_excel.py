import scipy
import scipy.io.wavfile
import os
import sys
import glob
import numpy as np


from python_speech_features import mfcc

from scipy import signal

import pywt
from pywt import dwt

import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile


from mysite.settings import *

from xlsite.models import Document



#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


'''
dictionaries where data shall be stored (in arrays) and then later used
in Pandas' DataFrame
'''

function_dict_1 = {} # for FFT 
function_dict_2 = {} # for MFCC  
function_dict_3 = {} # for STFT 
function_dict_4 = {} # for DWT  


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#FFT Part


def create_fft(wavfile): 
	'''
	Extracts frequencies from a wavile and stores in a file	
	here, fft_features is a single array
	'''

	sample_rate, song_array = scipy.io.wavfile.read(wavfile)
	#print(sample_rate)
	fft_features = abs(scipy.fft(song_array[:30000]))
	#print(song_array)
	#print(fft_features)
	base_fn, ext = os.path.splitext(wavfile)
	data_fn = base_fn + ".fft"
	#np.save(data_fn, fft_features)

	function_dict_1["FFT"] = fft_features




#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# MFCC Part 


def create_ceps(wavfile):
	"""
	Given a wavfile, computes mfcc and saves mfcc data
	Here, MFCC function (ceps) is a collection of arrays in an array.

	Get MFCC

	ceps  : ndarray of MFCC
	mspec : ndarray of log-spectrum in the mel-domain
	spec  : spectrum magnitude
	"""

	sampling_rate, song_array = scipy.io.wavfile.read(wavfile)
	#print(sampling_rate)

	ceps=mfcc(song_array[:30000])
	#ceps, mspec, spec= mfcc(song_array)
	#print(ceps.shape)
	#this is done in order to replace NaN and infinite value in array
	bad_indices = np.where(np.isnan(ceps))
	b=np.where(np.isinf(ceps))
	ceps[bad_indices]=0
	ceps[b]=0


	#inverting the array to fit data vertically into excel.
	rez = [[ceps[j][i] for j in range(len(ceps))] for i in range(len(ceps[0]))]
	
	ctr = 1
	for arr in rez :
		function_dict_2["MFCC " + str(ctr)] = arr
		ctr = ctr + 1
		



#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# STFT Part 


def create_stft_feature(wavfile):
	'''
	Here, STFT function (Zxx) is a collection of arrays in an array.
	'''

	sampling_rate, song_array = scipy.io.wavfile.read(wavfile)
	f , t ,Zxx = scipy.signal.stft(song_array[:30000],10e3,nperseg=1000)
	
	#print(Zxx.shape)

	bad_indices = np.where(np.isnan(Zxx))
	b=np.where(np.isinf(Zxx))
	Zxx[bad_indices]=0
	Zxx[b]=0
	#write_stft(Zxx, wavfile)


	#inverting the array to fit data vertically into excel.
	rez = [[Zxx[j][i] for j in range(len(Zxx))] for i in range(len(Zxx[0]))]
	
	ctr = 1
	for arr in rez :
		function_dict_3["STFT " + str(ctr)] = arr
		ctr = ctr + 1



#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# DWT Feature


def create_dwt_feature(wavfile):
	'''

	d_w_t is of this type :
	(arr[], arr[])

	'''
	sampling_rate, song_array = scipy.io.wavfile.read(wavfile)
	d_w_t = pywt.dwt(song_array[:30000],'db2')

	#inverting the array to fit data vertically into excel.
	#rez = [[d_w_t[j][i] for j in range(len(d_w_t))] for i in range(len(d_w_t[0]))]
	
	ctr = 1
	for arr in d_w_t :
		function_dict_4["DWT " + str(ctr)] = arr
		ctr = ctr + 1


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



def main_function(docName):

	'''

	docName is the string which tells the name of the file to be downloaded.

	Returns :

	1. Success message
	2. full file location (to download later) 

	'''


	for wavfile in os.listdir(MEDIA_ROOT):
		if wavfile.endswith("wav"):
			if wavfile == docName :  

				# for FFT Features :
				create_fft(os.path.join(MEDIA_ROOT + wavfile))


				# for MFCC Features :
				#create_ceps(os.path.join(MEDIA_ROOT + wavfile))


				# for STFT Features : 
				#create_stft_feature(os.path.join(MEDIA_ROOT + wavfile))


				# for DWT Features : 
				#create_dwt_feature(os.path.join(MEDIA_ROOT + wavfile))


				
				# for FFT :
				df_feature_1 = pd.DataFrame(function_dict_1)
				
				# for MFCC : 
				df_feature_2 = pd.DataFrame(function_dict_2)
				
				# for STFT : 
				df_feature_3 = pd.DataFrame(function_dict_3).astype(str) 
				
				# for DWT :
				df_feature_4 = pd.DataFrame(function_dict_4)


				# combining them all : 
				df = pd.concat([df_feature_1, df_feature_2, df_feature_3, df_feature_4], ignore_index=False, axis=1) 

				#print(df.head())


				# to get the file name. 
				base_fn, ext = os.path.splitext(wavfile)
				fname = base_fn + ".xlsx"
				
				full_file_loc = MEDIA_ROOT + 'convFiles/' + fname

				writer = ExcelWriter(full_file_loc)
				df.to_excel(writer,'Sheet1',index=False)
				writer.save()

					
				return("File " + fname + " created!", full_file_loc)

				

				
