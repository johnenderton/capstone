from django import forms
from django.core import validators

iterative_imputer_estimators = {
    ('BayesianRidge', 'BayesianRidge'),
    ('DecisionTreeRegressor', 'DecisionTreeRegressor'),
    ('ExtraTreesRegressor', 'ExtraTreesRegressor'),
    ('KNeighborsRegressor', 'KNeighborsRegressor'),
    ('DecisionTreeClassifier', 'DecisionTreeClassifier')
}


class Iterative_Imputer_Form(forms.Form):
    estimator = forms.ChoiceField(
        widget=forms.Select(),
        choices=iterative_imputer_estimators,
        initial='BayesianRidge'
    )
    # sample_posterior = forms.BooleanField(
    #     widget=forms.CheckboxInput,
    #     initial=False,
    #     required=False
    # )
    max_iter = forms.IntegerField(
        initial=10,
        widget=forms.NumberInput()
    )
    tol = forms.FloatField(
        initial=1e-3,
        widget=forms.NumberInput()
    )
    n_nearest_feature = forms.IntegerField(
        initial=None,
        required=False,
        widget=forms.NumberInput()
    )
    initial_strategy = forms.ChoiceField(
        widget=forms.Select,
        choices={
            ('mean', 'Mean'),
            ('median', 'Median'),
            ('most_frequent', 'Most Frequent'),
            ('constant', 'Constant'),
        }
    )
    imputation_order = forms.ChoiceField(
        widget=forms.Select(),
        choices={
            ('ascending', 'Ascending'),
            ('descending', 'Descending'),
            ('roman', 'Roman'),
            ('arabic', 'Arabic'),
            ('random', 'Random')
        }
    )
    skip_complete = forms.BooleanField(
        widget=forms.CheckboxInput,
        initial=False,
        required=False
    )
    min_value = forms.FloatField(
        initial=None,
        required=False,
        widget=forms.NumberInput()
    )
    max_value = forms.FloatField(
        initial=None,
        required=False,
        widget=forms.NumberInput()
    )
    verbose = forms.IntegerField(
        initial=0,
        widget=forms.NumberInput()
    )
    random_state = forms.IntegerField(
        initial=None,
        required=False,
        widget=forms.NumberInput()
    )

    # add_indicator = forms.BooleanField(
    #     widget=forms.CheckboxInput,
    #     initial=False,
    #     required=False
    # )

    def __init__(self, *args, **kwargs):
        super(Iterative_Imputer_Form, self).__init__(*args, **kwargs)
        self.fields['imputation_order'].initial = 'ascending'
        self.fields['initial_strategy'].initial = 'most_frequent'


class Simple_Imputer_Form(forms.Form):
    feature_type = forms.ChoiceField(
        widget=forms.Select(attrs={'id': 'simple_imputer_feature_type'}),
        choices={
            ('all', 'All'),
            ('numeric_features', 'Numeric Features'),
            ('categorical_features', 'Categorical Features')
        },
        label='Feature Type'
    )
    strategy = forms.ChoiceField(
        widget=forms.Select(attrs={'id': 'simple_imputer_strategy'}),
        choices={
            ('mean', 'Mean'),
            ('median', 'Median'),
            ('most_frequent', 'Most Frequent'),
            ('constant', 'Constant'),
        }
    )
    fill_value = forms.CharField(
        widget=forms.NumberInput(attrs={'id': 'simple_imputer_fill_value'}),
        initial=None,
        required=False
    )
    verbose = forms.IntegerField(
        initial=0
    )

    # copy = forms.BooleanField(
    #     initial=True,
    #     required=False
    # )
    # add_indicator = forms.BooleanField(
    #     initial=False,
    #     required=False
    # )

    def clean(self):
        all_clean_data = super(Simple_Imputer_Form, self).clean()
        strategy = all_clean_data['strategy']
        fill_value = all_clean_data['fill_value']

        if strategy == 'constant':
            if (fill_value is None) | (fill_value == ''):
                raise forms.ValidationError('A Constant Strategy Needs a Constant Value. Please Enter a Value in '
                                            '"fill_value" Field!')
