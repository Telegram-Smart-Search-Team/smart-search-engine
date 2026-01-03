from django.urls import path

from . import views


urlpatterns = [
    path("", views.index, name="index"),
    path("api/bootstrap", views.api_bootstrap, name="api_bootstrap"),
    path("api/batch", views.api_batch, name="api_batch"),
    path("api/chat-config", views.api_chat_config, name="api_chat_config"),
]
