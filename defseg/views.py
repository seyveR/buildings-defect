from django.shortcuts import render
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from django.http import HttpResponse, HttpRequest
import ultralytics
from ultralytics import YOLO
from ultralytics.engine.results import Results
import dill
from django.views.decorators.csrf import csrf_exempt
import os
from django.conf import settings
import base64
from io import BytesIO
from math import ceil
import torch
import random
from django.core.files.uploadedfile import InMemoryUploadedFile

model = YOLO(model="defseg/yolov8s.pt")

def home(request: HttpRequest):
    if request.method == 'POST' and request.FILES['photo']:
        # Получаем загруженное изображение из запроса
        photo = request.FILES['photo']
        
        image: Image = Image.open(photo)

        results: list[Results] = model.predict(image, iou=0.5, conf=0.15)
        
        for result in results:
            width, height = result.masks.shape[-2:]
           
            single_mask_area = result.masks.data.any(dim=0).sum().item()
            total_image_area = width * height
            print(total_image_area)
            print('single_mask_area', single_mask_area / total_image_area)

        # mask_area_percentage = ceil((single_mask_area / total_image_area) * 100 * 100) / 100
        mask_area_percentage = f'{(single_mask_area / total_image_area) * 100:.2f}'

        buffered_original = BytesIO()
        for chunk in photo.chunks():
            buffered_original.write(chunk)
        
        # Конвертируем содержимое оригинального изображения в строку base64
        original_image_content = base64.b64encode(buffered_original.getvalue()).decode('utf-8')

        buffered = BytesIO()
        lable_target = []

        for result in results:
            for box in result.boxes:
                lable_target.append({'class': result.names[box.cls.item()],
                                    'conf': box.conf.item()})
            annot = result.plot(boxes=False, labels=True, conf=True, line_width=2)
            im = Image.fromarray(annot[..., ::-1])  # RGB PIL image
            im.save('processed_image.jpg')
            im.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        lable_target_out = set()
        for item in sorted(lable_target, key=lambda x: x['conf'], reverse=True):
            lable_target_out.add(item['class'])


        context = {'lable_target': ', '.join(lable_target_out),
                   'output_image_path': img_str,
                   'def_area': mask_area_percentage,
                   'original_image_content': original_image_content
                   }
        return render(request, 'home.html', context)
    
    return render(request, 'home.html') 