from django.urls import path
from .views import PestDetectionView  # ✅ correct import

urlpatterns = [
    path('predict/', PestDetectionView.as_view(), name='predict'),
]
