from django.core.files.base import ContentFile
from django.http import HttpResponseRedirect
from django.shortcuts import render
from django.core.paginator import EmptyPage, PageNotAnInteger
from pure_pagination.paginator import Paginator
import os
from django.http import Http404
from .forms.index_form import UploadFileForm
from .forms.encoders_form import *
from .forms.missing_imputation_form import Simple_Imputer_Form, Iterative_Imputer_Form
from .forms.plot_graphs import *
from .forms.feature_scaling_form import *
from .data_cleaning import PPD
import pandas as pd
import seaborn as sns
import numpy as np
from bootstrap_modal_forms.generic import BSModalCreateView
from django.urls import reverse_lazy

# Plotly
from plotly.offline import plot
import plotly.graph_objects as pgo
from plotly.graph_objects import Box
import plotly.express as px
import plotly.io as pio

# AWS
import boto3

# From custom templatetags
from .templatetags import proper_paginate

# Create your views here.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_files_dir = None
file_name = None
ppd = PPD()
AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')


def index(request):
    print("Index function")
    global ppd
    global data_files_dir
    global file_name
    folder = "data_files"
    try:
        os.mkdir(os.path.join(BASE_DIR, folder))
    except:
        pass

    data_files_dir = os.path.join(BASE_DIR, 'data_files')
    if request.method == 'POST':

        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():

            # Get full url of the file
            # full_file_name = os.path.join(data_files_dir, str(request.FILES['Upload_File']))
            # fout = open(full_file_name, 'wb+')
            # file_content = ContentFile(request.FILES['Upload_File'].read())
            # try:
            #     # Iterate through the chunks.
            #     for chunk in file_content.chunks():
            #         fout.write(chunk)
            #     fout.close()
            #     print("Success")
            # except:
            #     print("Fail")
            #
            # # Get file name
            # file_name = form.cleaned_data['Upload_File']
            # file_name = os.path.join(data_files_dir, str(file_name))
            # ppd.init(file_name)
            data = open(str(form.cleaned_data['Upload_File']), 'rb')

            s3 = boto3.resource(
                's3',
                aws_access_key_id=AWS_ACCESS_KEY_ID,
                aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            )
            s3.Bucket('django-capstone').put_object(
                Key=str(form.cleaned_data['Upload_File']),
                Body=data
            )
            ppd.init(str(form.cleaned_data['Upload_File']))

            return HttpResponseRedirect('data_cleaning_app/status/')
    else:
        form = UploadFileForm()
    return render(request, 'data_cleaning_app/index.html', {'form': form})


def status(request):
    global file_name
    global ppd
    print("status function")

    if request.method == 'POST':
        if 'btn_status_remove_feature' in request.POST:
            feature = request.POST['btn_status_remove_feature']
            print("Remove ")
            print(feature)
            ppd.remove_feature(feature)

    feature_status = pd.DataFrame(ppd.get_feature_status())
    feature_status_dict = {}
    feature_status_columns = feature_status.axes[1]

    for i in feature_status.axes[0]:
        feature_status_dict[i] = {
            'attr_type': feature_status.at[i, 'attr_type'],
            'n_unique_val': feature_status.at[i, 'n_unique_val'],
            'number_of_missing': feature_status.at[i, 'number_of_missing'],
            'percentage_of_missing': feature_status.at[i, 'percentage_of_missing'],
            'mean': feature_status.at[i, 'Mean'],
            'median': feature_status.at[i, 'Median'],
            'variance': feature_status.at[i, 'Variance'],
            'standard_deviation': feature_status.at[i, 'Standard Deviation'],
            'category_stat': ppd.get_category_stat(i) if feature_status.at[i, 'attr_type'] == 'category' else None
        }

    # Pagination
    page = request.GET.get('page', 1)
    page_size = request.GET.get('page_size', 10)
    try:
        page = int(page)
        page_size = int(page_size)

    except ValueError:
        raise Http404
    status_paginator = Paginator(tuple(feature_status_dict.items()), page_size, request=request)
    try:
        feature_status_page = status_paginator.page(page)
    except PageNotAnInteger:
        feature_status_page = status_paginator.page(1)
    except EmptyPage:
        feature_status_page = status_paginator.page(status_paginator.num_pages)

    num_features_types = ppd.get_num_numeric_and_category_feature()
    num_missing = ppd.get_num_missing()
    context = {
        'feature_status': feature_status_page,
        'feature_status_columns': feature_status_columns,
        'num_page': status_paginator.num_pages,
        'num_rows': ppd.getDataShape()[0],
        'num_cols': ppd.getDataShape()[1],
        'num_numeric_features': num_features_types[0],
        'num_category_features': num_features_types[1],
        'num_missing': num_missing[0],
        'percent_missing': num_missing[1]
    }
    return render(request, 'data_cleaning_app/status.html', context=context)


def data_cleaning(request):
    print("data_cleaning function:")
    global ppd
    if request.method == 'POST':
        print("Post Data")
        print(request.POST)
        if 'binary_btn' in request.POST:
            binary_form = BinaryEncoder(request.POST)
            if binary_form.is_valid():
                ppd.binary_encode(
                    binary_form.cleaned_data['feature_name'],
                    binary_form.cleaned_data['verbose'],
                    binary_form.cleaned_data['drop_invariant'],
                    binary_form.cleaned_data['return_df'],
                    binary_form.cleaned_data['handle_unknown'],
                    binary_form.cleaned_data['handle_missing'],
                )
                print("Binary Encoder")
                print(binary_form.cleaned_data['feature_name'])
                print("Success")

        if 'hashing_btn' in request.POST:
            hashing_form = HashingEncoder(request.POST)
            if hashing_form.is_valid():
                ppd.hashing_encoder(
                    hashing_form.cleaned_data['feature_name'],
                    hashing_form.cleaned_data['verbose'],
                    hashing_form.cleaned_data['drop_invariant'],
                    hashing_form.cleaned_data['return_df'],
                    hashing_form.cleaned_data['hash_method'],
                    hashing_form.cleaned_data['max_process'],
                    hashing_form.cleaned_data['max_sample']
                )
                print("Hashing Encoder")
                print(hashing_form.cleaned_data['feature_name'])
                print("Hashing Encoder Form Success")

        if 'leave_one_out_btn' in request.POST:
            leave_one_out_form = LeaveOneOutEncoder(request.POST)
            if leave_one_out_form.is_valid():
                ppd.leave_one_out_encode(
                    leave_one_out_form.cleaned_data['feature_name'],
                    leave_one_out_form.cleaned_data['verbose'],
                    leave_one_out_form.cleaned_data['drop_invariant'],
                    leave_one_out_form.cleaned_data['return_df'],
                    leave_one_out_form.cleaned_data['handle_unknown'],
                    leave_one_out_form.cleaned_data['handle_missing'],
                    leave_one_out_form.cleaned_data['sigma'],
                )
                print("Leave One Out Encoder")
                print(leave_one_out_form.cleaned_data['feature_name'])
                print("Leave One Out Encoder Form Success")

        if 'one_hot_btn' in request.POST:
            one_hot_form = OneHotEncoder(request.POST)
            if one_hot_form.is_valid():
                ppd.one_hot_encode(
                    one_hot_form.cleaned_data['feature_name'],
                    one_hot_form.cleaned_data['drop'],
                    one_hot_form.cleaned_data['sparse'],
                    one_hot_form.cleaned_data['dType'],
                    one_hot_form.cleaned_data['handle_unknown']
                )
                print("One hot Encoder")
                print(request.POST)
                print("One Hot Encoder Form Success")

        if 'sum_btn' in request.POST:
            sum_form = SumEncoder(request.POST)
            if sum_form.is_valid():
                ppd.sum_encode(
                    sum_form.cleaned_data['feature_name'],
                    sum_form.cleaned_data['verbose'],
                    sum_form.cleaned_data['drop_invariant'],
                    sum_form.cleaned_data['return_df'],
                    sum_form.cleaned_data['handle_unknown'],
                    sum_form.cleaned_data['handle_missing']
                )
                print("Sum Encoder")
                print(sum_form.cleaned_data['feature_name'])
                print("Sum Encoder Form Success")

        if 'target_btn' in request.POST:
            target_form = TargetEncoder(request.POST)
            if target_form.is_valid():
                ppd.target_encoder(
                    target_form.cleaned_data['feature_name'],
                    target_form.cleaned_data['verbose'],
                    target_form.cleaned_data['drop_invariant'],
                    target_form.cleaned_data['return_df'],
                    target_form.cleaned_data['handle_unknown'],
                    target_form.cleaned_data['min_sample_leaf'],
                    target_form.cleaned_data['smoothing']
                )
                print("Target Encoder")
                print(target_form.cleaned_data['feature_name'])
                print("Target Encoder Form Success")

        if 'label_btn' in request.POST:
            label_form = LabelEncoder(request.POST)
            if label_form.is_valid():
                ppd.auto_label_encoding(label_form.cleaned_data['feature_name'])

    auto_form = AutoEncoderForm()
    manual_form = ManualEncoderForm()
    one_hot = OneHotEncoder()
    binary = BinaryEncoder()
    hashing = HashingEncoder()
    sum_encoder = SumEncoder()
    target = TargetEncoder()
    leave_one_out = LeaveOneOutEncoder()
    label = LabelEncoder()

    # Pagination
    page = request.GET.get('page', 1)
    page_size = request.GET.get('page_size', 10)
    try:
        page = int(page)
        page_size = int(page_size)

    except ValueError:
        raise Http404

    paginator = Paginator(tuple(ppd.get_category_list().T.items()), page_size, request=request)

    try:
        feature_name_page = paginator.page(page)
    except PageNotAnInteger:
        feature_name_page = paginator.page(1)
    except EmptyPage:
        feature_name_page = paginator.page(paginator.num_pages)

    context = {
        'auto_form': auto_form,
        'manual_form': manual_form,
        'category_feature_name': feature_name_page,
        'num_page': paginator.num_pages,
        'label': label,
        'one_hot': one_hot,
        'binary': binary,
        'hashing': hashing,
        'sum': sum_encoder,
        'target': target,
        'leave_one_out': leave_one_out,
        'have_missing': ppd.check_any_missing()
    }

    return render(request, 'data_cleaning_app/data_cleaning.html', context=context)


def missing_imputation(request):
    print("missing_imputation function")
    if request.method == 'POST':
        if 'simple_imputer_btn' in request.POST:
            simple_imputer = Simple_Imputer_Form(request.POST)

            if simple_imputer.is_valid():
                print("Simple Imputer Parameters")
                print(request.POST)
                ppd.simple_imputer(
                    simple_imputer.cleaned_data['feature_type'],
                    simple_imputer.cleaned_data['strategy'],
                    simple_imputer.cleaned_data['fill_value'],
                    simple_imputer.cleaned_data['verbose'],
                    # simple_imputer.cleaned_data['copy'],
                    # simple_imputer.cleaned_data['add_indicator']
                )

        if 'iterative_imputer_btn' in request.POST:
            iterative_imputer = Iterative_Imputer_Form(request.POST)
            if iterative_imputer.is_valid():
                print("Iterative Imputer Parameters")
                print(request.POST)
                ppd.iterative_imputer(
                    iterative_imputer.cleaned_data['estimator'],
                    # iterative_imputer.cleaned_data['sample_posterior'],
                    iterative_imputer.cleaned_data['max_iter'],
                    iterative_imputer.cleaned_data['tol'],
                    iterative_imputer.cleaned_data['n_nearest_feature'],
                    iterative_imputer.cleaned_data['initial_strategy'],
                    iterative_imputer.cleaned_data['imputation_order'],
                    iterative_imputer.cleaned_data['skip_complete'],
                    iterative_imputer.cleaned_data['min_value'],
                    iterative_imputer.cleaned_data['max_value'],
                    iterative_imputer.cleaned_data['verbose'],
                    iterative_imputer.cleaned_data['random_state'],
                    # iterative_imputer.cleaned_data['add_indicator']
                )
                print("transfer success")
        return HttpResponseRedirect('data_cleaning_app/status/')
    else:
        simple_imputer = Simple_Imputer_Form()
        iterative_imputer = Iterative_Imputer_Form()
        context = {
            'simple_imputer': simple_imputer,
            'iterative_imputer': iterative_imputer
        }
    return render(request, 'data_cleaning_app/missing_imputation.html', context=context)


def view_data(request):
    global ppd
    print("View data function")
    data = pd.DataFrame(ppd.get_data())

    # Pagination
    page = request.GET.get('page', 1)
    page_size = request.GET.get('page_size', 10)

    try:
        page = int(page)
        page_size = int(page_size)

    except ValueError:
        raise Http404

    view_data_paginator = Paginator(tuple(data.T.items()), page_size, request=request)
    try:
        data_status_page = view_data_paginator.page(page)
    except PageNotAnInteger:
        print("Page not an integer")
        data_status_page = view_data_paginator.page(1)
    except EmptyPage:
        print("empty page")
        data_status_page = view_data_paginator.page(view_data_paginator.num_pages)

    context = {
        'data': data_status_page,
        'features_name': ppd.getFeatureName(),
        'num_page': view_data_paginator.num_pages
    }
    return render(request, 'data_cleaning_app/view_data.html', context=context)


def outliers(request):
    global ppd
    print("outliers function")
    if request.method == 'POST':
        print("POST data")
        print(request.POST['outlier'])
        ppd.remove_feature_outlier_data(request.POST['outlier'])

    numeric_data = ppd.get_numeric_data()
    numeric_features_name = ppd.get_numeric_features_name()
    feature_box_plot = {}

    # Calculate Quartile
    # ppd.cal_quartile()

    for i in numeric_features_name:
        fig = px.box(numeric_data.loc[numeric_data[i].notnull(), i], y=i, points='all', width=600)
        feature_box_plot[i] = {
            'box_plot': pio.to_html(fig=fig, full_html=False, include_plotlyjs=False),
            'num_outlier': ppd.get_feature_num_outlier(i),
            'have_missing': ppd.check_feature_missing(i)
        }

    context = {
        'feature_box_plot': feature_box_plot
    }
    return render(request, 'data_cleaning_app/outliers.html', context=context)


def plot_graphs(request):
    print("plot_graphs function")
    global ppd
    fig = None
    fig_error = False

    blank_choice = (None, '---------')
    features_name = [(i, i) for i in ppd.getFeatureName()]
    category_features_name = [(i, i) for i in ppd.get_category_list()]
    numeric_features_name = [(i, i) for i in ppd.get_numeric_features_name()]
    features_name.append(blank_choice)
    category_features_name.append(blank_choice)
    numeric_features_name.append(blank_choice)

    if request.method == 'POST':
        if 'scatter_btn' in request.POST:
            print("scatter form")
            print(request.POST)
            scatter = Scatter_form(request.POST)
            scatter.fields['x'].choices = features_name
            scatter.fields['y'].choices = features_name
            scatter.fields['facet_row'].choices = category_features_name
            scatter.fields['facet_col'].choices = category_features_name
            scatter.fields['color'].choices = category_features_name
            scatter.fields['size'].choices = numeric_features_name
            print(scatter.errors)

            if scatter.is_valid():
                print("form valid")
                x = scatter.cleaned_data['x']
                y = scatter.cleaned_data['y']
                facet_row = scatter.cleaned_data['facet_row']
                facet_col = scatter.cleaned_data['facet_col']
                facet_col_wrap = scatter.cleaned_data['facet_col_wrap']
                color = scatter.cleaned_data['color']
                size = scatter.cleaned_data['size']

                data_feature_list = list()
                data_feature_list.append(x)
                data_feature_list.append(y)
                if len(size) > 0:
                    data_feature_list.append(size)
                if len(color) > 0:
                    data_feature_list.append(color)
                if len(facet_row) > 0:
                    data_feature_list.append(facet_row)
                if len(facet_col) > 0:
                    data_feature_list.append(facet_col)

                data = pd.DataFrame(ppd.get_features_data(data_feature_list))
                try:
                    fig = px.scatter(
                        data_frame=data,
                        x=x,
                        y=y,
                        facet_row=None if len(facet_row) == 0 else facet_row,
                        facet_col=None if len(facet_col) == 0 else facet_col,
                        facet_col_wrap=facet_col_wrap,
                        title=scatter.cleaned_data['title'],
                        color=None if len(color) == 0 else color,
                        size=None if len(size) == 0 else size,
                        log_x=scatter.cleaned_data['log_x'],
                        log_y=scatter.cleaned_data['log_y'],
                        render_mode=scatter.cleaned_data['render_mode'],
                        height=800
                    )
                    print("fig create success")
                except:
                    print("fig create error")
                    fig_error = True
                    fig = None

        if 'scatter_3d_btn' in request.POST:
            scatter_3d = Scatter_3d_form(request.POST)
            scatter_3d.fields['x'].choices = features_name
            scatter_3d.fields['y'].choices = features_name
            scatter_3d.fields['z'].choices = features_name
            scatter_3d.fields['color'].choices = category_features_name
            scatter_3d.fields['size'].choices = numeric_features_name

            if scatter_3d.is_valid():
                x = scatter_3d.cleaned_data['x']
                y = scatter_3d.cleaned_data['y']
                z = scatter_3d.cleaned_data['z']
                color = scatter_3d.cleaned_data['color']
                size = scatter_3d.cleaned_data['size']

                data_feature_list = list()
                data_feature_list.append(x)
                data_feature_list.append(y)
                data_feature_list.append(z)
                if len(size) > 0:
                    data_feature_list.append(size)
                if len(color) > 0:
                    data_feature_list.append(color)

                data = pd.DataFrame(ppd.get_features_data(data_feature_list))

                try:
                    fig = px.scatter_3d(
                        data_frame=data,
                        x=x,
                        y=y,
                        z=z,
                        title=scatter_3d.cleaned_data['title'],
                        color=None if len(color) == 0 else color,
                        size=None if len(size) == 0 else size,
                        log_x=scatter_3d.cleaned_data['log_x'],
                        log_y=scatter_3d.cleaned_data['log_y'],
                        log_z=scatter_3d.cleaned_data['log_z'],
                        height=800
                    )
                    print("fig create success")
                except:
                    print("fig create error")
                    fig_error = True
                    fig = None

        if 'line_btn' in request.POST:
            print("Line Plot")
            line = Line_form(request.POST)
            line.fields['x'].choices = features_name
            line.fields['y'].choices = features_name
            line.fields['facet_row'].choices = category_features_name
            line.fields['facet_col'].choices = category_features_name
            line.fields['color'].choices = category_features_name

            if line.is_valid():
                print("line is valid")
                x = line.cleaned_data['x']
                y = line.cleaned_data['y']
                facet_row = line.cleaned_data['facet_row']
                facet_col = line.cleaned_data['facet_col']
                facet_col_wrap = line.cleaned_data['facet_col_wrap']
                color = line.cleaned_data['color']

                data_feature_list = list()
                data_feature_list.append(x)
                data_feature_list.append(y)

                if len(color) > 0:
                    data_feature_list.append(color)
                if len(facet_row) > 0:
                    data_feature_list.append(facet_row)
                if len(facet_col) > 0:
                    data_feature_list.append(facet_col)

                data = pd.DataFrame(ppd.get_features_data(data_feature_list))

                try:
                    fig = px.line(
                        data_frame=data,
                        x=x,
                        y=y,
                        facet_row=None if len(facet_row) == 0 else facet_row,
                        facet_col=None if len(facet_col) == 0 else facet_col,
                        facet_col_wrap=facet_col_wrap,
                        title=line.cleaned_data['title'],
                        color=None if len(color) == 0 else color,
                        height=800
                    )
                    print("fig create success")
                except:
                    print("fig create error")
                    fig_error = True
                    fig = None

        if 'bar_btn' in request.POST:
            print("Bar Plot")
            bar = Bar_form(request.POST)
            bar.fields['x'].choices = features_name
            bar.fields['y'].choices = features_name
            bar.fields['facet_row'].choices = category_features_name
            bar.fields['facet_col'].choices = category_features_name
            bar.fields['color'].choices = category_features_name

            if bar.is_valid():
                print("Bar is valid")
                x = bar.cleaned_data['x']
                y = bar.cleaned_data['y']
                facet_row = bar.cleaned_data['facet_row']
                facet_col = bar.cleaned_data['facet_col']
                facet_col_wrap = bar.cleaned_data['facet_col_wrap']
                color = bar.cleaned_data['color']
                title = bar.cleaned_data['title']
                orientation = bar.cleaned_data['orientation']
                bar_mode = bar.cleaned_data['bar_mode']

                data_feature_list = list()
                data_feature_list.append(x)
                data_feature_list.append(y)

                if len(color) > 0:
                    data_feature_list.append(color)
                if len(facet_row) > 0:
                    data_feature_list.append(facet_row)
                if len(facet_col) > 0:
                    data_feature_list.append(facet_col)

                data = pd.DataFrame(ppd.get_features_data(data_feature_list))

                try:
                    fig = px.bar(
                        data_frame=data,
                        x=x,
                        y=y,
                        facet_row=None if len(facet_row) == 0 else facet_row,
                        facet_col=None if len(facet_col) == 0 else facet_col,
                        facet_col_wrap=facet_col_wrap,
                        title=title,
                        color=None if len(color) == 0 else color,
                        orientation=orientation,
                        barmode=bar_mode,
                        height=800
                    )
                    print("fig create success")
                except:
                    print("fig create error")
                    fig_error = True
                    fig = None

        if 'pie_btn' in request.POST:
            print("Pie Plot")
            pie = Pie_form(request.POST)
            pie.fields['values'].choices = features_name
            pie.fields['names'].choices = category_features_name
            pie.fields['color'].choices = category_features_name

            if pie.is_valid():
                print("Pie is valid")
                values = pie.cleaned_data['values']
                names = pie.cleaned_data['names']
                color = pie.cleaned_data['color']
                title = pie.cleaned_data['title']

                data_feature_list = list()
                data_feature_list.append(values)
                data_feature_list.append(names)

                if len(color) > 0:
                    data_feature_list.append(color)

                data = pd.DataFrame(ppd.get_features_data(data_feature_list))

                try:
                    fig = px.pie(
                        data_frame=data,
                        values=values,
                        names=names,
                        color=None if len(color) == 0 else color,
                        title=title,
                        height=800
                    )
                    print("fig create success")
                except:
                    print("fig create error")
                    fig_error = True
                    fig = None

        if 'histogram_btn' in request.POST:
            print("Plot Histogram")
            histogram = Histogram_form(request.POST)
            histogram.fields['x'].choices = features_name
            histogram.fields['y'].choices = features_name
            histogram.fields['facet_row'].choices = category_features_name
            histogram.fields['facet_col'].choices = category_features_name
            histogram.fields['color'].choices = category_features_name
            print(histogram.errors)

            if histogram.is_valid():
                print("Histogram is valid")
                x = histogram.cleaned_data['x']
                y = histogram.cleaned_data['y']
                facet_row = histogram.cleaned_data['facet_row']
                facet_col = histogram.cleaned_data['facet_col']
                facet_col_wrap = histogram.cleaned_data['facet_col_wrap']
                color = histogram.cleaned_data['color']
                title = histogram.cleaned_data['title']
                orientation = histogram.cleaned_data['orientation']
                bar_mode = histogram.cleaned_data['bar_mode']
                marginal = histogram.cleaned_data['marginal']
                bar_norm = histogram.cleaned_data['bar_norm']
                hist_norm = histogram.cleaned_data['hist_norm']
                hist_func = histogram.cleaned_data['hist_func']
                log_x = histogram.cleaned_data['log_x']
                log_y = histogram.cleaned_data['log_y']
                cumulative = histogram.cleaned_data['cumulative']

                data_feature_list = list()
                data_feature_list.append(x)
                data_feature_list.append(y)

                if len(color) > 0:
                    data_feature_list.append(color)
                if len(facet_row) > 0:
                    data_feature_list.append(facet_row)
                if len(facet_col) > 0:
                    data_feature_list.append(facet_col)

                data = pd.DataFrame(ppd.get_features_data(data_feature_list))

                try:
                    fig = px.histogram(
                        data_frame=data,
                        x=x,
                        y=y,
                        facet_row=None if len(facet_row) == 0 else facet_row,
                        facet_col=None if len(facet_col) == 0 else facet_col,
                        facet_col_wrap=facet_col_wrap,
                        title=title,
                        color=None if len(color) == 0 else color,
                        orientation=orientation,
                        barmode=bar_mode,
                        marginal=marginal,
                        barnorm=bar_norm,
                        histnorm=hist_norm,
                        histfunc=hist_func,
                        log_x=log_x,
                        log_y=log_y,
                        cumulative=cumulative,
                        height=800
                    )
                    print("Fig create success")
                except:
                    print("fig create error")
                    fig_error = True
                    fig = None

        if 'scatter_matrix_btn' in request.POST:
            print("Scatter Matrix Plot")
            scatter_matrix = Scatter_matrix_form(request.POST)
            scatter_matrix.fields['feature_1'].choices = numeric_features_name
            scatter_matrix.fields['feature_2'].choices = numeric_features_name
            scatter_matrix.fields['feature_3'].choices = numeric_features_name
            scatter_matrix.fields['feature_4'].choices = numeric_features_name
            scatter_matrix.fields['color'].choices = category_features_name
            scatter_matrix.fields['size'].choices = numeric_features_name
            scatter_matrix.fields['symbol'].choices = category_features_name

            if scatter_matrix.is_valid():
                print("Scatter Matrix is valid")
                feature_1 = scatter_matrix.cleaned_data['feature_1']
                feature_2 = scatter_matrix.cleaned_data['feature_2']
                feature_3 = scatter_matrix.cleaned_data['feature_3']
                feature_4 = scatter_matrix.cleaned_data['feature_4']
                color = scatter_matrix.cleaned_data['color']
                symbol = scatter_matrix.cleaned_data['symbol']
                size = scatter_matrix.cleaned_data['size']
                title = scatter_matrix.cleaned_data['title']

                data_feature_list = list()
                data_feature_list.append(feature_1)
                data_feature_list.append(feature_2)
                data_feature_list.append(feature_3)
                data_feature_list.append(feature_4)
                if len(size) > 0:
                    data_feature_list.append(size)
                if len(color) > 0:
                    data_feature_list.append(color)

                data = pd.DataFrame(ppd.get_features_data(data_feature_list))

                try:
                    fig = px.scatter_matrix(
                        data_frame=data,
                        dimensions=[feature_1, feature_2, feature_3, feature_4],
                        color=None if len(color) == 0 else color,
                        symbol=None if len(symbol) == 0 else symbol,
                        size=None if len(size) == 0 else size,
                        title=title
                    )
                    print("fig create success")
                except:
                    print("fig create error")
                    fig_error = True
                    fig = None

        if 'box_btn' in request.POST:
            box = Box_form(request.POST)
            box.fields['x'].choices = features_name
            box.fields['y'].choices = features_name
            box.fields['facet_row'].choices = category_features_name
            box.fields['facet_col'].choices = category_features_name
            box.fields['color'].choices = category_features_name
            print(box.errors)

            if box.is_valid():
                x = box.cleaned_data['x']
                y = box.cleaned_data['y']
                facet_row = box.cleaned_data['facet_row']
                facet_col = box.cleaned_data['facet_col']
                color = box.cleaned_data['color']
                facet_col_wrap = box.cleaned_data['facet_col_wrap']
                title = box.cleaned_data['title']
                orientation = box.cleaned_data['orientation']
                log_x = box.cleaned_data['log_x']
                log_y = box.cleaned_data['log_y']
                box_mode = box.cleaned_data['box_mode']
                points = box.cleaned_data['points']
                notched = box.cleaned_data['notched']

                data_feature_list = list()
                if len(x) > 0:
                    data_feature_list.append(x)
                if len(y) > 0:
                    data_feature_list.append(y)
                if len(color) > 0:
                    data_feature_list.append(color)
                if len(facet_row) > 0:
                    data_feature_list.append(facet_row)
                if len(facet_col) > 0:
                    data_feature_list.append(facet_col)

                data = pd.DataFrame(ppd.get_features_data(data_feature_list))

                try:
                    fig = px.box(
                        data_frame=data,
                        x=None if len(x) == 0 else x,
                        y=None if len(y) == 0 else y,
                        facet_row=None if len(facet_row) == 0 else facet_row,
                        facet_col=None if len(facet_col) == 0 else facet_col,
                        facet_col_wrap=facet_col_wrap,
                        title=title,
                        orientation=orientation,
                        log_x=log_x,
                        log_y=log_y,
                        boxmode=box_mode,
                        points=points,
                        notched=notched
                    )
                    print("fig create success")
                except:
                    print("fig create error")
                    fig_error = True
                    fig = None

        if 'violin_btn' in request.POST:
            violin = Violin_form(request.POST)
            violin.fields['x'].choices = features_name
            violin.fields['y'].choices = features_name
            violin.fields['facet_row'].choices = category_features_name
            violin.fields['facet_col'].choices = category_features_name
            violin.fields['color'].choices = category_features_name
            print(violin.errors)

            if violin.is_valid():
                x = violin.cleaned_data['x']
                y = violin.cleaned_data['y']
                facet_row = violin.cleaned_data['facet_row']
                facet_col = violin.cleaned_data['facet_col']
                color = violin.cleaned_data['color']
                facet_col_wrap = violin.cleaned_data['facet_col_wrap']
                title = violin.cleaned_data['title']
                orientation = violin.cleaned_data['orientation']
                log_x = violin.cleaned_data['log_x']
                log_y = violin.cleaned_data['log_y']
                violin_mode = violin.cleaned_data['violin_mode']
                points = violin.cleaned_data['points']
                box = violin.cleaned_data['box']

                data_feature_list = list()
                if len(x) > 0:
                    data_feature_list.append(x)
                if len(y) > 0:
                    data_feature_list.append(y)
                if len(color) > 0:
                    data_feature_list.append(color)
                if len(facet_row) > 0:
                    data_feature_list.append(facet_row)
                if len(facet_col) > 0:
                    data_feature_list.append(facet_col)

                data = pd.DataFrame(ppd.get_features_data(data_feature_list))

                try:
                    fig = px.violin(
                        data_frame=data,
                        x=None if len(x) == 0 else x,
                        y=None if len(y) == 0 else y,
                        facet_row=None if len(facet_row) == 0 else facet_row,
                        facet_col=None if len(facet_col) == 0 else facet_col,
                        facet_col_wrap=facet_col_wrap,
                        title=title,
                        orientation=orientation,
                        log_x=log_x,
                        log_y=log_y,
                        violinmode=violin_mode,
                        points=points,
                        box=box
                    )
                    print("fig create success")
                except:
                    print("fig create error")
                    fig_error = True
                    fig = None

        if 'heat_map_btn' in request.POST:
            print("Heat Map")
            heat_map_data = pd.DataFrame(ppd.get_corr_matrix())
            print(heat_map_data.columns)
            try:
                fig = px.imshow(
                    heat_map_data.astype(float),
                    x=heat_map_data.columns,
                    y=heat_map_data.index,
                    zmax=1,
                    zmin=-1,
                    height=800
                )
            except:
                print("fig create error")
                fig_error = True
                fig = None

    scatter = Scatter_form()
    scatter.fields['x'].choices = features_name
    scatter.fields['y'].choices = features_name
    scatter.fields['facet_row'].choices = category_features_name
    scatter.fields['facet_col'].choices = category_features_name
    scatter.fields['color'].choices = category_features_name
    scatter.fields['size'].choices = numeric_features_name

    scatter_3d = Scatter_3d_form()
    scatter_3d.fields['x'].choices = features_name
    scatter_3d.fields['y'].choices = features_name
    scatter_3d.fields['z'].choices = features_name
    scatter_3d.fields['color'].choices = category_features_name
    scatter_3d.fields['size'].choices = numeric_features_name

    line = Line_form()
    line.fields['x'].choices = features_name
    line.fields['y'].choices = features_name
    line.fields['facet_row'].choices = category_features_name
    line.fields['facet_col'].choices = category_features_name
    line.fields['color'].choices = category_features_name

    bar = Bar_form()
    bar.fields['x'].choices = features_name
    bar.fields['y'].choices = features_name
    bar.fields['facet_row'].choices = category_features_name
    bar.fields['facet_col'].choices = category_features_name
    bar.fields['color'].choices = category_features_name

    pie = Pie_form()
    pie.fields['values'].choices = features_name
    pie.fields['names'].choices = category_features_name
    pie.fields['color'].choices = category_features_name

    histogram = Histogram_form()
    histogram.fields['x'].choices = features_name
    histogram.fields['y'].choices = features_name
    histogram.fields['facet_row'].choices = category_features_name
    histogram.fields['facet_col'].choices = category_features_name
    histogram.fields['color'].choices = category_features_name

    scatter_matrix = Scatter_matrix_form()
    scatter_matrix.fields['feature_1'].choices = numeric_features_name
    scatter_matrix.fields['feature_2'].choices = numeric_features_name
    scatter_matrix.fields['feature_3'].choices = numeric_features_name
    scatter_matrix.fields['feature_4'].choices = numeric_features_name
    scatter_matrix.fields['color'].choices = category_features_name
    scatter_matrix.fields['size'].choices = numeric_features_name
    scatter_matrix.fields['symbol'].choices = category_features_name

    box = Box_form()
    box.fields['x'].choices = features_name
    box.fields['y'].choices = features_name
    box.fields['facet_row'].choices = category_features_name
    box.fields['facet_col'].choices = category_features_name
    box.fields['color'].choices = category_features_name

    violin = Violin_form()
    violin.fields['x'].choices = features_name
    violin.fields['y'].choices = features_name
    violin.fields['facet_row'].choices = category_features_name
    violin.fields['facet_col'].choices = category_features_name
    violin.fields['color'].choices = category_features_name

    context = {
        'fig':  None,
        'scatter': scatter,
        'line': line,
        'scatter_3d': scatter_3d,
        'bar': bar,
        'pie': pie,
        'histogram': histogram,
        'scatter_matrix': scatter_matrix,
        'box': box,
        'violin': violin
    }
    if fig is not None:
        context['fig'] = pio.to_html(fig=fig, full_html=False, include_plotlyjs=False)
    elif fig_error is True:
        context['fig'] = "Plot Graph Error When Setting Parameters. Please Try Again!"
    else:
        context['fig'] = None
    return render(request, 'data_cleaning_app/plot_graphs.html', context=context)


def feature_scaling(request):
    global ppd
    if request.method == 'POST':
        print("feature scaling post")
        if 'normalization' in request.POST:
            ppd.set_normalization('all')

        if 'standardization' in request.POST:
            ppd.set_standardization('all')

        if 'scaling_btn' in request.POST:
            scaling_feature = Feature_Scaling_form(request.POST)
            print(scaling_feature.errors)
            if scaling_feature.is_valid():
                print(request.POST)
                print(request.POST['scaling_btn'])
                scaling_method = scaling_feature.cleaned_data['scaling_method']
                if scaling_method == 'normalization':
                    ppd.set_normalization(request.POST['scaling_btn'])
                else:
                    ppd.set_standardization(request.POST['scaling_btn'])

    # Pagination
    page = request.GET.get('page', 1)
    page_size = request.GET.get('page_size', 10)
    try:
        page = int(page)
        page_size = int(page_size)

    except ValueError:
        raise Http404
    paginator = Paginator(tuple(ppd.get_numeric_features_name()), page_size, request=request)

    try:
        feature_name_page = paginator.page(page)
    except PageNotAnInteger:
        feature_name_page = paginator.page(1)
    except EmptyPage:
        feature_name_page = paginator.page(paginator.num_pages)

    scaling_feature = Feature_Scaling_form()

    context = {
        'scaling_feature': scaling_feature,
        'feature_name': feature_name_page,
        'num_page': paginator.num_pages,
        'have_missing': ppd.check_any_missing()
    }
    return render(request, 'data_cleaning_app/feature_scaling.html', context=context)


def correlation_matrix(request):
    global ppd
    corr = pd.DataFrame(ppd.get_corr_matrix())
    feature_name = ppd.getFeatureName()
    corr.insert(0, column='Feature Name', value=feature_name)

    # Pagination
    page = request.GET.get('page', 1)
    page_size = request.GET.get('page_size', 12)
    try:
        page = int(page)
        page_size = int(page_size)

    except ValueError:
        raise Http404
    paginator = Paginator(tuple(corr.T.items()), page_size, request=request)

    try:
        corr_page = paginator.page(page)
    except PageNotAnInteger:
        corr_page = paginator.page(1)
    except EmptyPage:
        corr_page = paginator.page(paginator.num_pages)

    context = {
        'corr': corr_page,
        'features_name': ppd.getFeatureName(),
    }
    return render(request, 'data_cleaning_app/correlation_matrix.html', context=context)
