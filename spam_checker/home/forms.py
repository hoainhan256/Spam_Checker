# home/forms.py
from django import forms

class EmailForm(forms.Form):
    email_text = forms.CharField(widget=forms.Textarea(attrs={'rows': 5}), label='Nhập nội dung email')
