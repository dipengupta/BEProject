# from django import forms

# # class UploadFileForm(forms.Form):
# #     title = forms.CharField(max_length=50)
# #     file = forms.FileField()


# class DocumentForm(forms.Form):
#     docfile = forms.FileField(
#         label='Select a file',
#         help_text='max. 42 megabytes'
#     )


from django import forms

class DocumentForm(forms.Form):
    docfile = forms.FileField(
        label='Select a file',
        help_text='[NOTE : ".wav" file only, max 42 megabytes]'
    )




genre_list= [
    (1, 'Blues'),
    (2, 'Classical'),
    (3, 'Country'),
    (4, 'Disco'),
    (5, 'Classical'),
    (6, 'Jazz'),
    (7, 'Metal'),
    (8, 'Pop'),
    (9, 'Reggae'),
    (10, 'Rock')
]



class UserForm(forms.Form):
    selected_genre = forms.CharField(label='Select a genre to convert to : ', widget=forms.Select(choices=genre_list))