from django.conf.urls import url
from django.urls import path
from data_cleaning_app import views

urlpatterns = [
    url(r'^$', views.index, name='index'),
    url(r'^status', views.status, name='status'),
    url(r'^data_cleaning', views.data_cleaning, name='data_cleaning'),
    url(r'^missing_imputation/', views.missing_imputation, name='missing_imputation'),
    url(r'^view_data', views.view_data, name='view_data'),
    url(r'^outliers/', views.outliers, name='outliers'),
    url(r'^plot_graphs/', views.plot_graphs, name='plot_graphs'),
    url(r'^feature_scaling', views.feature_scaling, name='feature_scaling'),
    url(r'^correlation_matrix/', views.correlation_matrix, name='correlation_matrix')
    # path('one_hot/', views.OneHotEncoderView.as_view, name='one_hot')
]
