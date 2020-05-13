from django import forms


class UploadFileForm(forms.Form):
    Upload_File = forms.FileField(required=True)
