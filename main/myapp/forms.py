from django import forms
from django.contrib.auth.forms import UserCreationForm
from myapp.models import CustomUser

class RegisterForm(forms.Form):
    firstname = forms.CharField(max_length=50)
    lastname = forms.CharField(max_length=50)
    email = forms.EmailField()
    password = forms.CharField(widget=forms.PasswordInput)
