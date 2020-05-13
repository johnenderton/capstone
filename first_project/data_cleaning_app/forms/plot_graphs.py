from django import forms
from django.core import validators


def get_choices(features_name):
    temp = ['']
    for i in features_name:
        temp.append(i)

    return temp


class Scatter_form(forms.Form):

    x = forms.ChoiceField(
        widget=forms.Select(attrs={'id': 'scatter_x'}),
        label='x',
        initial=None
    )
    y = forms.ChoiceField(
        widget=forms.Select(attrs={'id': 'scatter_y'}),
        label='y',
        initial=None
    )
    facet_row = forms.ChoiceField(
        widget=forms.Select(attrs={'id': 'scatter_facet_row'}),
        label='Facet_row',
        initial=None,
        required=False
    )
    facet_col = forms.ChoiceField(
        widget=forms.Select(attrs={'id': 'scatter_facet_col'}),
        label='Facet_col',
        initial=None,
        required=False
    )
    facet_col_wrap = forms.IntegerField(
        widget=forms.NumberInput,
        label='Facet_col_wrap',
        initial=None,
        required=False
    )
    color = forms.ChoiceField(
        widget=forms.Select(attrs={'id': 'scatter_color'}),
        initial=None,
        label='Color',
        required=False
    )
    title = forms.CharField(
        widget=forms.TextInput,
        label='Title',
        required=False
    )
    size = forms.ChoiceField(
        widget=forms.Select(attrs={'id': 'scatter_size'}),
        initial=None,
        label='Size',
        required=False
    )
    log_x = forms.BooleanField(
        widget=forms.CheckboxInput(attrs={'id': 'scatter_log_x'}),
        initial=False,
        label='Log_x',
        required=False
    )
    log_y = forms.BooleanField(
        widget=forms.CheckboxInput(attrs={'id': 'scatter_log_y'}),
        initial=False,
        label='Log_y',
        required=False
    )
    render_mode = forms.ChoiceField(
        widget=forms.Select(attrs={'id': 'scatter_render_mode'}),
        initial='auto',
        choices={
            ('auto', 'Auto'),
            ('svg', 'SVG'),
            ('webgl', 'WebGL')
        },
        label='Render Mode',
        required=False
    )

    # def clean(self):
    #     all_clean_data = super().clean()
    #     x = all_clean_data['x']
    #     y = all_clean_data['y']
    #     facet_row = all_clean_data['facet_row']
    #     facet_col = all_clean_data['facet_col']
    #
    #     if (x is None) | (x == ''):
    #         raise forms.ValidationError('Field "x" cannot be empty!')
    #
    #     if (y is None) | (y == ''):
    #         raise forms.ValidationError('Field "y" cannot be empty!')
    #
    #     if facet_row is not None:
    #         if (facet_row != x) | (facet_row != y):
    #             raise forms.ValidationError('Facet_row must be either x or y')
    #
    #     if facet_col is not None:
    #         if (facet_col != x) | (facet_col != y):
    #             raise forms.ValidationError('Facet_col must be either x or y')
    #
    #     if (facet_row == x) | (facet_row == y):
    #         if facet_row == facet_col:
    #             raise forms.ValidationError('Can Only Choose One Facet')
    #
    #     if (facet_col == x) | (facet_col == y):
    #         if facet_col == facet_row:
    #             raise forms.ValidationError('Can Only Choose One Facet')


class Scatter_3d_form(forms.Form):

    x = forms.ChoiceField(
        widget=forms.Select(attrs={'id': 'scatter_3d_x'}),
        label='x',
        initial=None
    )
    y = forms.ChoiceField(
        widget=forms.Select(attrs={'id': 'scatter_3d_y'}),
        label='y',
        initial=None
    )
    z = forms.ChoiceField(
        widget=forms.Select(attrs={'id': 'scatter_3d_z'}),
        label='z',
        initial=None
    )
    color = forms.ChoiceField(
        widget=forms.Select(attrs={'id': 'scatter_3d_color'}),
        initial=None,
        label='Color',
        required=False
    )
    title = forms.CharField(
        widget=forms.TextInput,
        label='Title',
        required=False
    )
    size = forms.ChoiceField(
        widget=forms.Select(attrs={'id': 'scatter_3d_size'}),
        initial=None,
        label='Size',
        required=False
    )
    log_x = forms.BooleanField(
        widget=forms.CheckboxInput(attrs={'id': 'scatter_3d_log_x'}),
        initial=False,
        label='Log x',
        required=False
    )
    log_y = forms.BooleanField(
        widget=forms.CheckboxInput(attrs={'id': 'scatter_3d_log_y'}),
        initial=False,
        label='Log y',
        required=False
    )
    log_z = forms.BooleanField(
        widget=forms.CheckboxInput(attrs={'id': 'scatter_3d_log_z'}),
        initial=False,
        label='Log z',
        required=False
    )

    # def clean(self):
    #     all_clean_data = super().clean()
    #     x = all_clean_data['x']
    #     y = all_clean_data['y']
    #     z = all_clean_data['z']
    #
    #     if x is None:
    #         raise forms.ValidationError('Field "x" cannot be empty!')
    #
    #     if y is None:
    #         raise forms.ValidationError('Field "y" cannot be empty!')
    #
    #     if z is None:
    #         raise forms.ValidationError('Field "z" cannot be empty!')


class Line_form(forms.Form):
    x = forms.ChoiceField(
        widget=forms.Select(attrs={'id': 'line_x'}),
        label='x',
        initial=None
    )
    y = forms.ChoiceField(
        widget=forms.Select(attrs={'id': 'line_y'}),
        label='y',
        initial=None
    )
    facet_row = forms.ChoiceField(
        widget=forms.Select(attrs={'id': 'line_facet_row'}),
        label='Facet_row',
        initial=None,
        required=False
    )
    facet_col = forms.ChoiceField(
        widget=forms.Select(attrs={'id': 'line_facet_col'}),
        label='Facet_col',
        initial=None,
        required=False
    )
    facet_col_wrap = forms.IntegerField(
        widget=forms.NumberInput,
        label='Facet_col_wrap',
        initial=None,
        required=False
    )
    color = forms.ChoiceField(
        widget=forms.Select(attrs={'id': 'line_color'}),
        initial=None,
        label='Color',
        required=False
    )
    title = forms.CharField(
        widget=forms.TextInput,
        label='Title',
        required=False
    )

    # def clean(self):
    #     all_clean_data = super().clean()
    #     x = all_clean_data['x']
    #     y = all_clean_data['y']
    #     facet_row = all_clean_data['facet_row']
    #     facet_col = all_clean_data['facet_col']
    #
    #     if x is None:
    #         raise forms.ValidationError('Field "x" cannot be empty!')
    #
    #     if y is None:
    #         raise forms.ValidationError('Field "y" cannot be empty!')
    #
    #     if facet_row is not None:
    #         if (facet_row != x) | (facet_row != y):
    #             raise forms.ValidationError('Facet_row must be either x or y')
    #
    #     if facet_col is not None:
    #         if (facet_col != x) | (facet_col != y):
    #             raise forms.ValidationError('Facet_col must be either x or y')
    #
    #     if (facet_row == x) | (facet_row == y):
    #         if facet_row == facet_col:
    #             raise forms.ValidationError('Can Only Choose One Facet')
    #
    #     if (facet_col == x) | (facet_col == y):
    #         if facet_col == facet_row:
    #             raise forms.ValidationError('Can Only Choose One Facet')


class Bar_form(forms.Form):
    x = forms.ChoiceField(
        widget=forms.Select(attrs={'id': 'bar_x'}),
        label='x',
        initial=None
    )
    y = forms.ChoiceField(
        widget=forms.Select(attrs={'id': 'bar_y'}),
        label='y',
        initial=None
    )
    facet_row = forms.ChoiceField(
        widget=forms.Select(attrs={'id': 'bar_facet_row'}),
        label='Facet_row',
        initial=None,
        required=False
    )
    facet_col = forms.ChoiceField(
        widget=forms.Select(attrs={'id': 'bar_facet_col'}),
        label='Facet_col',
        initial=None,
        required=False
    )
    facet_col_wrap = forms.IntegerField(
        widget=forms.NumberInput,
        label='Facet_col_wrap',
        initial=None,
        required=False
    )
    color = forms.ChoiceField(
        widget=forms.Select(attrs={'id': 'bar_color'}),
        initial=None,
        label='Color',
        required=False
    )
    title = forms.CharField(
        widget=forms.TextInput,
        label='Title',
        required=False
    )
    orientation = forms.ChoiceField(
        widget=forms.Select(attrs={'id': 'bar_orientation'}),
        initial='v',
        choices={('v', 'Vertical'), ('h', 'Horizontal')},
        label='Orientation',
        required=False
    )
    bar_mode = forms.ChoiceField(
        widget=forms.Select(attrs={'id': 'bar_mode'}),
        initial='relative',
        choices={('relative', 'Relative'), ('group', 'Group'), ('overlay', 'Overlay')},
        label='Bar Mode',
        required=False
    )

    # def clean(self):
    #     all_clean_data = super().clean()
    #     x = all_clean_data['x']
    #     y = all_clean_data['y']
    #     facet_row = all_clean_data['facet_row']
    #     facet_col = all_clean_data['facet_col']
    #
    #     if x is None:
    #         raise forms.ValidationError('Field "x" cannot be empty!')
    #
    #     if y is None:
    #         raise forms.ValidationError('Field "y" cannot be empty!')
    #
    #     if facet_row is not None:
    #         if (facet_row != x) | (facet_row != y):
    #             raise forms.ValidationError('Facet_row must be either x or y')
    #
    #     if facet_col is not None:
    #         if (facet_col != x) | (facet_col != y):
    #             raise forms.ValidationError('Facet_col must be either x or y')
    #
    #     if (facet_row == x) | (facet_row == y):
    #         if facet_row == facet_col:
    #             raise forms.ValidationError('Can Only Choose One Facet')
    #
    #     if (facet_col == x) | (facet_col == y):
    #         if facet_col == facet_row:
    #             raise forms.ValidationError('Can Only Choose One Facet')


class Pie_form(forms.Form):
    values = forms.ChoiceField(
        widget=forms.Select(attrs={'id': 'pie_values'}),
        initial=None,
        label='Values'
    )
    names = forms.ChoiceField(
        widget=forms.Select(attrs={'id': 'pie_names'}),
        initial=None,
        label='Names'
    )
    color = forms.ChoiceField(
        widget=forms.Select(attrs={'id': 'pie_color'}),
        initial=None,
        label='Color',
        required=False
    )
    title = forms.CharField(
        widget=forms.TextInput,
        label='Title',
        required=False
    )


class Histogram_form(forms.Form):
    x = forms.ChoiceField(
        widget=forms.Select(attrs={'id': 'histogram_x'}),
        label='x',
        initial=None
    )
    y = forms.ChoiceField(
        widget=forms.Select(attrs={'id': 'histogram_y'}),
        label='y',
        initial=None
    )
    facet_row = forms.ChoiceField(
        widget=forms.Select(attrs={'id': 'histogram_facet_row'}),
        label='Facet_row',
        initial=None,
        required=False
    )
    facet_col = forms.ChoiceField(
        widget=forms.Select(attrs={'id': 'histogram_facet_col'}),
        label='Facet_col',
        initial=None,
        required=False
    )
    facet_col_wrap = forms.IntegerField(
        widget=forms.NumberInput,
        label='Facet_col_wrap',
        initial=None,
        required=False
    )
    color = forms.ChoiceField(
        widget=forms.Select(attrs={'id': 'histogram_color'}),
        initial=None,
        label='Color',
        required=False
    )
    title = forms.CharField(
        widget=forms.TextInput,
        label='Title',
        required=False
    )
    marginal = forms.ChoiceField(
        widget=forms.Select,
        initial=None,
        choices={
            (None, '-----------'),
            ('rug', 'Rug'),
            ('box', 'Box'),
            ('violin', 'Violin'),
            ('histogram', 'Histogram')
        },
        required=False
    )
    orientation = forms.ChoiceField(
        widget=forms.Select,
        initial='v',
        choices={
            ('v', 'Vertical'),
            ('h', 'Horizontal')
        },
        required=False
    )
    bar_mode = forms.ChoiceField(
        widget=forms.Select,
        initial='relative',
        choices={
            ('relative', 'Relative'),
            ('group', 'Group'),
            ('overlay', 'Overlay')
        },
        required=False
    )
    bar_norm = forms.ChoiceField(
        widget=forms.Select,
        initial=None,
        choices={
            (None, '------------'),
            ('fraction', 'Fraction'),
            ('percent', 'Percent')
        },
        required=False
    )
    hist_norm = forms.ChoiceField(
        widget=forms.Select,
        initial=None,
        choices={
            (None, '------------'),
            ('percent', 'Percent'),
            ('probability', 'Probability'),
            ('density', 'Density'),
            ('probability density', 'probability Density')
        },
        required=False
    )
    hist_func = forms.ChoiceField(
        widget=forms.Select,
        initial='count',
        choices={
            ('count', 'Count'),
            ('sum', 'Sum'),
            ('avg', 'Avg'),
            ('min', 'Min'),
            ('max', 'Max')
        },
        required=False
    )
    log_x = forms.BooleanField(
        widget=forms.CheckboxInput,
        initial=False,
        required=False
    )
    log_y = forms.BooleanField(
        widget=forms.CheckboxInput,
        initial=False,
        required=False
    )
    cumulative = forms.BooleanField(
        widget=forms.CheckboxInput,
        initial=False,
        required=False
    )


class Scatter_matrix_form(forms.Form):
    feature_1 = forms.ChoiceField(
        widget=forms.Select(attrs={'id': 'scatter_matrix_feature_1'}),
        initial=None,
        label='Feature 1'
    )
    feature_2 = forms.ChoiceField(
        widget=forms.Select(attrs={'id': 'scatter_matrix_feature_2'}),
        initial=None,
        label='Feature 2'
    )
    feature_3 = forms.ChoiceField(
        widget=forms.Select(attrs={'id': 'scatter_matrix_feature_3'}),
        initial=None,
        label='Feature 3'
    )
    feature_4 = forms.ChoiceField(
        widget=forms.Select(attrs={'id': 'scatter_matrix_feature_4'}),
        initial=None,
        label='Feature 4'
    )
    color = forms.ChoiceField(
        widget=forms.Select(attrs={'id': 'scatter_matrix_color'}),
        initial=None,
        label='Color',
        required=False
    )
    symbol = forms.ChoiceField(
        widget=forms.Select(attrs={'id': 'scatter_matrix_symbol'}),
        initial=None,
        label='Symbol',
        required=False
    )
    size = forms.ChoiceField(
        widget=forms.Select(attrs={'id': 'scatter_matrix_size'}),
        initial=None,
        label='Size',
        required=False
    )
    title = forms.CharField(
        widget=forms.TextInput,
        label='Title',
        required=False
    )


class Box_form(forms.Form):
    x = forms.ChoiceField(
        widget=forms.Select(attrs={'id': 'box_x'}),
        label='x',
        initial=None,
        required=False
    )
    y = forms.ChoiceField(
        widget=forms.Select(attrs={'id': 'box_y'}),
        label='y',
        initial=None,
        required=False
    )
    facet_row = forms.ChoiceField(
        widget=forms.Select(attrs={'id': 'box_facet_row'}),
        label='Facet_row',
        initial=None,
        required=False
    )
    facet_col = forms.ChoiceField(
        widget=forms.Select(attrs={'id': 'box_facet_col'}),
        label='Facet_col',
        initial=None,
        required=False
    )
    facet_col_wrap = forms.IntegerField(
        widget=forms.NumberInput,
        label='Facet_col_wrap',
        initial=None,
        required=False
    )
    color = forms.ChoiceField(
        widget=forms.Select(attrs={'id': 'box_color'}),
        initial=None,
        label='Color',
        required=False
    )
    title = forms.CharField(
        widget=forms.TextInput,
        label='Title',
        required=False
    )
    orientation = forms.ChoiceField(
        widget=forms.Select,
        initial='v',
        choices={
            ('v', 'Vertical'),
            ('h', 'Horizontal')
        },
        required=False
    )
    log_x = forms.BooleanField(
        widget=forms.CheckboxInput,
        initial=False,
        required=False
    )
    log_y = forms.BooleanField(
        widget=forms.CheckboxInput,
        initial=False,
        required=False
    )
    box_mode = forms.ChoiceField(
        widget=forms.Select,
        initial='group',
        choices={
            ('group', 'Group'),
            ('overlay', 'Overlay')
        },
        required=False
    )
    points = forms.ChoiceField(
        widget=forms.Select,
        initial='outliers',
        choices={
            ('outliers', 'Outliers'),
            ('suspectedoutliers', 'Suspectedoutliers'),
            ('all', 'All'),
            ('False', 'False')
        },
        required=False
    )
    notched = forms.BooleanField(
        widget=forms.CheckboxInput,
        initial=False,
        required=False
    )


class Violin_form(forms.Form):

    x = forms.ChoiceField(
        widget=forms.Select(attrs={'id': 'violin_x'}),
        label='x',
        initial=None,
        required=False
    )
    y = forms.ChoiceField(
        widget=forms.Select(attrs={'id': 'violin_y'}),
        label='y',
        initial=None,
        required=False
    )
    facet_row = forms.ChoiceField(
        widget=forms.Select(attrs={'id': 'violin_facet_row'}),
        label='Facet_row',
        initial=None,
        required=False
    )
    facet_col = forms.ChoiceField(
        widget=forms.Select(attrs={'id': 'violin_facet_col'}),
        label='Facet_col',
        initial=None,
        required=False
    )
    facet_col_wrap = forms.IntegerField(
        widget=forms.NumberInput,
        label='Facet_col_wrap',
        initial=None,
        required=False
    )
    color = forms.ChoiceField(
        widget=forms.Select(attrs={'id': 'violin_color'}),
        initial=None,
        label='Color',
        required=False
    )
    title = forms.CharField(
        widget=forms.TextInput,
        label='Title',
        required=False
    )
    orientation = forms.ChoiceField(
        widget=forms.Select,
        initial='v',
        choices={
            ('v', 'Vertical'),
            ('h', 'Horizontal')
        },
        required=False
    )
    log_x = forms.BooleanField(
        widget=forms.CheckboxInput,
        initial=False,
        required=False
    )
    log_y = forms.BooleanField(
        widget=forms.CheckboxInput,
        initial=False,
        required=False
    )
    violin_mode = forms.ChoiceField(
        widget=forms.Select,
        initial='group',
        choices={
            ('group', 'Group'),
            ('overlay', 'Overlay')
        },
        required=False
    )
    points = forms.ChoiceField(
        widget=forms.Select,
        initial='outliers',
        choices={
            ('outliers', 'Outliers'),
            ('suspectedoutliers', 'Suspectedoutliers'),
            ('all', 'All'),
            ('False', 'False')
        },
        required=False
    )
    box = forms.BooleanField(
        widget=forms.CheckboxInput,
        initial=False,
        required=False
    )
