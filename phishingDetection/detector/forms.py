from django import forms

class MessageForm(forms.Form):
    # Define your form fields here
    '''subject = forms.CharField(max_length=255, required=True)
    sender = forms.EmailField(required=True)
    recipient = forms.EmailField(required=True)
    message = forms.CharField(widget=forms.Textarea, required=True)
    url = forms.URLField(required=True)'''

    text = forms.CharField(widget=forms.Textarea(attrs={'class': 'form-control', 'placeholder': 'Enter your email contents or URL here... '}))