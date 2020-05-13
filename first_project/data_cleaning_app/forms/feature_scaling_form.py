from django import forms


class Feature_Scaling_form(forms.Form):
    scaling_method = forms.ChoiceField(
        widget=forms.Select(attrs={'class': 'manual_scaling'}),
        choices=(
            (None, '---------------'),
            ('normalization', 'Normalization'),
            ('standardization', 'Standardization')
        )
    )
