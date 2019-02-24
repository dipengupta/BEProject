from django.shortcuts import render
from django.http import HttpResponse
from django.http import HttpResponseRedirect
from django.template import RequestContext
from django.urls import reverse
from xlsite.models import Document
from xlsite.forms import DocumentForm
from mysite.settings import *
from xlsite.logic.export_features_to_excel import *
from xlsite.logic.classify import *
from xlsite.logic.convCode import *
import os
from os import walk



def download(request, path):
	'''
	function to download any media file from the path provided.
	'''
	file_path = path
	with open(file_path, 'rb') as fh:
		response = HttpResponse(fh.read(), content_type="application/vnd.ms-excel")
		response['Content-Disposition'] = 'inline; filename=' + os.path.basename(file_path)
	return response





def list(request):


	def cleanup(request) :
		'''
		~helper function~ : deletes all files present in media folder
		'''
		
		all_files = []
		for root,d_names,f_names in os.walk(MEDIA_ROOT):
			for f in f_names:
				all_files.append(os.path.join(root,f))

		#present_in_convFiles = []

		# for ele in all_files :
		# 	if 'convFiles' in ele :
		# 		present_in_convFiles.append(ele)
		# 	all_files.remove(ele)

		# for ele in all_files :
		# 	os.remove(ele)

		# for ele in present_in_convFiles :
		# 	os.remove(ele)	

		for ele in all_files :
			os.remove(ele)				



	def getNoOfFiles(request) :
		'''
		~helper function~ : returns no. of files present in media folder. 
		'''
		
		all_files = []
		for root,d_names,f_names in os.walk(MEDIA_ROOT):
			for f in f_names:
				all_files.append(os.path.join(root,f))

		return len(all_files)



	'''
	NOTE :

	Here, we can't just leave the last 2 elements out in hope that they are
	the latest ones uploaded (cause the files are stored alphabetically, not
	the most recent first)
	'''
	if(getNoOfFiles(request) > 32) :
		cleanup(request) 



	if request.method == 'POST':
		form = DocumentForm(request.POST, request.FILES)
		if form.is_valid():
			newdoc = Document(docfile = request.FILES['docfile'])
			newdoc.save()

			return HttpResponseRedirect(reverse('list'))
	else:
		form = DocumentForm()


	documents = Document.objects.all()

	
	return render(request, 'xlsite/list.html', {'documents': documents, 'form': form, 'xlData' : documents[::-1][0]})





def getFeatures(request) :

	temp_documents = Document.objects.all()[::-1]
	document = (temp_documents[0])

	docName = str(document.docfile.name)

	msg, filePath = main_function(docName)
	#return render(request,'xlsite/getFeatures.html',{'msg' : msg, 'filePath' : filePath})
	return download(request,filePath)
	




def seeFiles(request) :

	ans = []
	for root,d_names,f_names in os.walk(MEDIA_ROOT):
		for f in f_names:
			ans.append(os.path.join(root,f))

	return render(request, 'xlsite/seeFiles.html', {'documents': ans})





def findGenre(request) :

	temp_documents = Document.objects.all()[::-1]
	document = (temp_documents[0])

	docName = str(document.docfile.name)

	ans = plsWork(docName)

	#return HttpResponseRedirect(reverse('classify', kwargs={'genre': ans}))
	return render(request, 'xlsite/classify.html', {'genre': ans})



def convTo(request) :

	temp_documents = Document.objects.all()[::-1]
	document = (temp_documents[0])

	docName = str(document.docfile.name)

	userSelectedGenre = request.GET.get('convToGenre')
	ans = master_function(userSelectedGenre, docName)

	return download(request,ans)

	#return render(request, 'xlsite/convert.html', {'ans': ans})

