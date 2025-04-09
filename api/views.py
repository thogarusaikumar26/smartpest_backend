from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser
from .inference import predict_image
import tempfile

class PestDetectionView(APIView):
    parser_classes = [MultiPartParser]

    def post(self, request, format=None):
        image_file = request.FILES.get('image')
        if not image_file:
            return Response({"error": "No image uploaded."}, status=400)

        # Save image temporarily
        with tempfile.NamedTemporaryFile(delete=False) as temp:
            for chunk in image_file.chunks():
                temp.write(chunk)
            temp_path = temp.name

        # Predict
        result = predict_image(temp_path)
        return Response(result)
