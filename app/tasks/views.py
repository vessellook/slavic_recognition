import cv2 as cv
import numpy as np
from django.core.files.uploadedfile import UploadedFile
from django.http import HttpResponse
from drf_yasg import openapi
from drf_yasg.openapi import Parameter, Schema, TYPE_FILE
from drf_yasg.utils import swagger_auto_schema
from rest_framework.parsers import MultiPartParser
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView

from tasks.image_processing.howe import binarize as howe_binarize
from tasks.image_processing.profile_line_segmentation import get_point_lists, draw_dashes


class PingView(APIView):
    def get(self, request):
        return Response('pong')


class BinarizeView(APIView):
    parser_classes = [MultiPartParser]

    @swagger_auto_schema(
        manual_parameters=[
            Parameter(name='file', type=TYPE_FILE, in_=openapi.IN_FORM, required=True,
                      description='Изображение для бинаризации'),
        ],
        responses={
            200: openapi.Response(description='Бинаризованное изображение', schema=Schema(type=TYPE_FILE)),
            400: openapi.Response(description='Ошибка в запросе. Подробности в теле ответа'),
        },
    )
    def post(self, request: Request):
        file_obj: UploadedFile | None = request.FILES.get('file')
        if file_obj is None:
            return Response(status=400, data={'error': 'Нужно передать файл изображения'})
        input_file = file_obj.file
        image = cv.imdecode(np.frombuffer(input_file.read(), np.uint8), cv.IMREAD_COLOR)
        grayscale_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        binarized_image = howe_binarize(grayscale_image)
        output_file = cv.imencode('.png', binarized_image)[1].tobytes()
        mime_type = 'image/x-png'
        return HttpResponse(output_file, content_type=mime_type)


class DrawLineBordersView(APIView):
    parser_classes = [MultiPartParser]

    @swagger_auto_schema(
        manual_parameters=[
            Parameter(name='file', type=TYPE_FILE, in_=openapi.IN_FORM, required=True,
                      description='Изображение для сегментации строк'),
        ],
        responses={
            200: openapi.Response(description='Изображение с линиями сегментации строк', schema=Schema(type=TYPE_FILE)),
            400: openapi.Response(description='Ошибка в запросе. Подробности в теле ответа'),
        },
    )
    def post(self, request: Request):
        file_obj: UploadedFile | None = request.FILES.get('file')
        if file_obj is None:
            return Response(status=400, data={'error': 'Нужно передать файл изображения'})
        input_file = file_obj.file
        image = cv.imdecode(np.frombuffer(input_file.read(), np.uint8), cv.IMREAD_COLOR)
        grayscale_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        binarized_image = howe_binarize(grayscale_image)
        step = binarized_image.shape[1] // 150
        top_point_lists, bottom_point_lists = get_point_lists(255 - binarized_image, step)
        output_image = image.copy()
        print(top_point_lists, bottom_point_lists)
        draw_dashes(image=output_image, color=(255, 0, 0), points=sum(top_point_lists, []), step=step)
        # draw_dashes(image=output_image, color=(0, 255, 0), points=sum(bottom_point_lists, []), step=step)
        output_file = cv.imencode('.png', output_image)[1].tobytes()
        mime_type = 'image/x-png'
        return HttpResponse(output_file, content_type=mime_type)
