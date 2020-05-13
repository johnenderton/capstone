from django import forms
from bootstrap_modal_forms.forms import BSModalForm

# This choice work for both nominal and ordinal categories
encoder_choice = (
    ('label', 'Label Encoder'),
    ('one_hot', 'One Hot Encoder'),
    ('hashing', 'Hashing Encoder'),
    ('sum', 'Sum Encoder'),
    ('target', 'Target Encoder'),
    ('leave_one_out', 'Leave One Out Encoder')
)

# This one is for ordinal categories only
# Currently, this list has all encoders that currently in use
manual_encoder_choice = (
    (None, '-------------'),
    ('label', 'Label Encoding'),
    ('one_hot', 'One Hot Encoding'),
    ('binary', 'Binary Encoding'),
    ('hashing', 'Hashing Encoding'),
    ('sum', 'Sum Encoding'),
    ('target', 'Target Encoding'),
    ('leave_one_out', 'Leave One Out Encoding')
)

one_hot_choice = {
    ('error', 'error'), ('ignore', 'ignore')
}


class AutoEncoderForm(forms.Form):
    categories = forms.ChoiceField(choices=encoder_choice, widget=forms.RadioSelect)


class ManualEncoderForm(forms.Form):
    feature_name = forms.CharField(
        widget=forms.HiddenInput(attrs={'id': 'manual_encode_feature_name'}),
        initial='all'
    )
    category = forms.ChoiceField(choices=manual_encoder_choice, widget=forms.Select(attrs={'class': 'manual_encode'}))


class LabelEncoder(forms.Form):
    feature_name = forms.CharField(
        widget=forms.HiddenInput(attrs={'id': 'label_feature_name'}),
        label="",
        initial='all'
    )


class OneHotEncoder(forms.Form):
    feature_name = forms.CharField(
        widget=forms.HiddenInput(attrs={'id': 'one_hot_feature_name'}),
        label="",
        initial='all'
    )
    drop = forms.CharField(
        initial='first',
        label="Drop",
        widget=forms.TextInput(attrs={'id': 'one_hot_drop'}),
        required=False
    )
    sparse = forms.BooleanField(
        initial=True,
        label="Sparse",
        required=False,
        widget=forms.CheckboxInput(attrs={'id': 'one_hot_sparse'})
    )
    dType = forms.CharField(
        initial='np.float',
        label="dType",
        widget=forms.TextInput(attrs={'id': 'one_hot_dType'})
    )
    handle_unknown = forms.ChoiceField(
        choices=one_hot_choice,
        widget=forms.Select(attrs={'id': 'one_hot_handle_unknown'}),
        label="Handle Unknown"
    )

    def __init__(self, *args, **kwargs):
        super(OneHotEncoder, self).__init__(*args, **kwargs)
        self.fields['handle_unknown'].initial = 'error'


class BinaryEncoder(forms.Form):
    feature_name = forms.CharField(
        widget=forms.HiddenInput(attrs={'id': 'binary_feature_name'}),
        label="",
        initial='all'
    )
    verbose = forms.IntegerField(
        initial=0,
        label="Verbose",
        widget=forms.NumberInput(attrs={'id': 'binary_verbose'})
    )
    drop_invariant = forms.BooleanField(
        initial=False,
        label="Drop Invariant",
        required=False,
        widget=forms.CheckboxInput(attrs={'id': 'binary_drop_invariant'})
    )
    return_df = forms.BooleanField(
        initial=True,
        label="Return df",
        required=False,
        widget=forms.CheckboxInput(attrs={'id': 'binary_return_df'})
    )
    handle_unknown = forms.ChoiceField(
        choices={
            ('value', 'value'),
            ('error', 'error'),
            ('return_nan', 'return_nan'),
            ('indicator', 'indicator')
        },
        widget=forms.Select(attrs={'id': 'binary_handle_unknown'}),
        label="Handle Unknown"
    )
    handle_missing = forms.ChoiceField(
        choices={
            ('value', 'value'),
            ('error', 'error'),
            ('return_nan', 'return_nan'),
            ('indicator', 'indicator')
        },
        widget=forms.Select(attrs={'id': 'binary_handle_missing'}),
        label="Handle Missing"
    )

    def __init__(self, *args, **kwargs):
        super(BinaryEncoder, self).__init__(*args, **kwargs)
        self.fields['handle_unknown'].initial = 'value'
        self.fields['handle_missing'].initial = 'value'


class HashingEncoder(forms.Form):
    feature_name = forms.CharField(
        widget=forms.HiddenInput(attrs={'id': 'hashing_feature_name'}),
        label="",
        initial='all'
    )
    max_process = forms.IntegerField(
        initial=0,
        label="Max Process",
        widget=forms.NumberInput(attrs={'id': 'hashing_max_process'})
    )
    max_sample = forms.IntegerField(
        initial=0,
        label="Max Sample",
        widget=forms.NumberInput(attrs={'id': 'hashing_max_sample'})
    )
    verbose = forms.IntegerField(
        initial=0,
        label="Verbose",
        widget=forms.NumberInput(attrs={'id': 'hashing_verbose'})
    )
    drop_invariant = forms.BooleanField(
        initial=False,
        label="Drop Invariant",
        required=False,
        widget=forms.CheckboxInput(attrs={'id': 'hashing_drop_invariant'})
    )
    return_df = forms.BooleanField(
        initial=True,
        label="Return df",
        required=False,
        widget=forms.CheckboxInput(attrs={'id': 'hashing_drop_return_df'})
    )
    hash_method = forms.CharField(
        initial='md5',
        label="Hash Method",
        widget=forms.TextInput(attrs={'id': 'hashing_hash_method'})
    )


class SumEncoder(forms.Form):
    feature_name = forms.CharField(
        widget=forms.HiddenInput(attrs={'id': 'sum_feature_name'}),
        label="",
        initial='all'
    )
    verbose = forms.IntegerField(
        initial=0,
        label="Verbose",
        widget=forms.NumberInput(attrs={'id': 'sum_verbose'})
    )
    drop_invariant = forms.BooleanField(
        initial=False,
        label="Drop Invariant",
        required=False,
        widget=forms.CheckboxInput(attrs={'id': 'sum_drop_invariant'})
    )
    return_df = forms.BooleanField(
        initial=True,
        label="Return df",
        required=False,
        widget=forms.CheckboxInput(attrs={'id': 'sum_return_df'})
    )
    handle_unknown = forms.ChoiceField(
        choices={
            ('value', 'value'),
            ('error', 'error'),
            ('return_nan', 'return_nan'),
            ('indicator', 'indicator')
        },
        widget=forms.Select(attrs={'id': 'sum_handle_unknown'}),
        label="Handle Unknown"
    )
    handle_missing = forms.ChoiceField(
        choices={
            ('value', 'value'),
            ('error', 'error'),
            ('return_nan', 'return_nan'),
            ('indicator', 'indicator')
        },
        widget=forms.Select(attrs={'id': 'sum_handle_missing'}),
        label="Handle Missing"
    )

    def __init__(self, *args, **kwargs):
        super(SumEncoder, self).__init__(*args, **kwargs)
        self.fields['handle_unknown'].initial = 'value'
        self.fields['handle_missing'].initial = 'value'


class TargetEncoder(forms.Form):
    feature_name = forms.CharField(
        widget=forms.HiddenInput(attrs={'id': 'target_feature_name'}),
        label="",
        initial='all'
    )
    verbose = forms.IntegerField(
        initial=0,
        label="Verbose",
        widget=forms.NumberInput(attrs={'id': 'target_verbose'})
    )
    drop_invariant = forms.BooleanField(
        initial=False,
        label="Drop Invariant",
        required=False,
        widget=forms.CheckboxInput(attrs={'id': 'target_drop_invariant'})
    )
    return_df = forms.BooleanField(
        initial=True,
        label="Return df",
        required=False,
        widget=forms.CheckboxInput(attrs={'id': 'target_return_df'})
    )
    handle_unknown = forms.ChoiceField(
        choices={
            ('value', 'value'),
            ('error', 'error'),
            ('return_nan', 'return_nan'),
            ('indicator', 'indicator')
        },
        widget=forms.Select(attrs={'id': 'target_handle_unknown'}),
        label="Handle Unknown"
    )
    handle_missing = forms.ChoiceField(
        choices={
            ('value', 'value'),
            ('error', 'error'),
            ('return_nan', 'return_nan'),
            ('indicator', 'indicator')
        },
        widget=forms.Select(attrs={'id': 'target_handle_missing'}),
        label="Handle Missing"
    )
    min_sample_leaf = forms.IntegerField(
        initial=1,
        label="Min Sample Leaf",
        widget=forms.NumberInput(attrs={'id': 'target_min_sample_leaf'})
    )
    smoothing = forms.FloatField(
        initial=1.0,
        label="Smoothing",
        widget=forms.NumberInput(attrs={'id': 'target_smoothing'})
    )

    def __init__(self, *args, **kwargs):
        super(TargetEncoder, self).__init__(*args, **kwargs)
        self.fields['handle_unknown'].initial = 'value'
        self.fields['handle_missing'].initial = 'value'


class LeaveOneOutEncoder(forms.Form):
    feature_name = forms.CharField(
        widget=forms.HiddenInput(attrs={'id': 'leave_one_out_feature_name'}),
        label="",
        initial='all'
    )
    verbose = forms.IntegerField(
        initial=0,
        label="Verbose",
        widget=forms.NumberInput(attrs={'id': 'leave_one_out_verbose'})
    )
    drop_invariant = forms.BooleanField(
        initial=False,
        label="Drop Invariant",
        required=False,
        widget=forms.CheckboxInput(attrs={'id': 'leave_one_out_drop_invariant'})
    )
    return_df = forms.BooleanField(
        initial=True,
        label="Return df",
        required=False,
        widget=forms.CheckboxInput(attrs={'id': 'leave_one_out_return_df'})
    )
    handle_unknown = forms.ChoiceField(
        choices={
            ('value', 'value'),
            ('error', 'error'),
            ('return_nan', 'return_nan'),
            ('indicator', 'indicator')
        },
        widget=forms.Select(attrs={'id': 'leave_one_out_handle_unknown'}),
        label="Handle Missing"
    )
    handle_missing = forms.ChoiceField(
        choices={
            ('value', 'value'),
            ('error', 'error'),
            ('return_nan', 'return_nan'),
            ('indicator', 'indicator')
        },
        widget=forms.Select(attrs={'id': 'leave_one_out_handle_missing'}),
        label="Handle Missing"
    )
    sigma = forms.FloatField(
        initial=None,
        label="Sigma",
        widget=forms.NumberInput(attrs={'id': 'leave_one_out_sigma'}),
        required=False
    )

    def __init__(self, *args, **kwargs):
        super(LeaveOneOutEncoder, self).__init__(*args, **kwargs)
        self.fields['handle_unknown'].initial = 'value'
        self.fields['handle_missing'].initial = 'value'
