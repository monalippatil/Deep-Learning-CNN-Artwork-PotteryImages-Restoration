from django.db import models

import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanSquaredError as MSE
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import MaxPooling2D, UpSampling2D
from tensorflow.keras.layers import Conv2D
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model


from PIL import Image, ImageDraw
import random

import keras
from keras.utils import Sequence

# Add these:
from wagtail.models import Page
from wagtail.fields import RichTextField
from wagtail.admin.panels import FieldPanel

from wagtail.search import index

from wagtail.admin.panels import (
    FieldPanel,
    MultiFieldPanel
)
from django.shortcuts import render
# from wagtail.images.edit_handlers import ImageChooserPanel
# from wagtail.edit_handlers import ImageChooserPanel

from PIL import Image
import cv2
import uuid
import os
import glob
from pathlib import Path
from django.conf import settings
from django.core.files.storage import default_storage
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import random
from PIL import Image, ImageDraw

class BlogIndexPage(Page):
    intro = RichTextField(blank=True)

    content_panels = Page.content_panels + [
        FieldPanel('intro')
    ]

class BlogPage(Page):
    date = models.DateField("Post date")
    intro = models.CharField(max_length=250)
    body = RichTextField(blank=True)

    search_fields = Page.search_fields + [
        index.SearchField('intro'),
        index.SearchField('body'),
    ]

    content_panels = Page.content_panels + [
        FieldPanel('date'),
        FieldPanel('intro'),
        FieldPanel('body'),
    ]    

def draw_random_lines(input_file, output_file, num_lines=10, min_thickness=3, max_thickness=10):
    img = Image.open(input_file)
    img.thumbnail((256, 256), Image.LANCZOS)
    
    width, height = img.size
    draw = ImageDraw.Draw(img)
    
    for _ in range(num_lines):
        x1 = random.randint(0, width)
        y1 = random.randint(0, height)
        x2 = random.randint(0, width)
        y2 = random.randint(0, height)
        
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        thickness = random.randint(min_thickness, max_thickness)
        
        draw.line([(x1, y1), (x2, y2)], fill=color, width=thickness)
    
    img.save(output_file)


def reset():
    files_result = glob.glob(
        str(Path(f"{settings.MEDIA_ROOT}/Result/*.*")), recursive=True
    )
    files_upload = glob.glob(
        str(Path(f"{settings.MEDIA_ROOT}/uploadedPics/*.*")), recursive=True
    )
    files_processed = glob.glob(
        str(Path(f"{settings.MEDIA_ROOT}/processed_images/*.*")), recursive=True
    )
    files = []
    if len(files_result) != 0:
        files.extend(files_result)
    if len(files_upload) != 0:
        files.extend(files_upload)
    if len(files_processed) != 0:
        files.extend(files_processed)
    if len(files) != 0:
        for f in files:
            try:
                if not (f.endswith(".txt")):
                    os.remove(f)
            except OSError as e:
                print("Error: %s : %s" % (f, e.strerror))
        file_li = [
            Path(f"{settings.MEDIA_ROOT}/Result/Result.txt"),
            Path(f"{settings.MEDIA_ROOT}/uploadedPics/img_list.txt"),
            Path(f"{settings.MEDIA_ROOT}/processed_images/img_list.txt"),
            Path(f"{settings.MEDIA_ROOT}/Result/stats.txt"),
        ]
        for p in file_li:
            file = open(Path(p), "r+")
            file.truncate(0)
            file.close()

def dice_coef(y_true, y_pred):
    y_true_f = keras.backend.flatten(y_true)
    y_pred_f = keras.backend.flatten(y_pred)
    intersection = keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection) / (keras.backend.sum(y_true_f + y_pred_f))

def iou(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
    union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou


# Load the U-Net model#
MODEL_PATH = os.path.join(settings.MODEL_ROOT, "UNet-CNN-model.h5") # Adjust the path to your model
unet_model = load_model(MODEL_PATH, custom_objects={"dice_coef": dice_coef})


def process_and_predict(image_path, model):
    """Process an uploaded image and predict using the U-Net model."""
    
    img = load_img(image_path, target_size=(256, 256))  # Adjust size to your model's expected input
    img_array = img_to_array(img) / 255.0
    prediction = model.predict(np.expand_dims(img_array, axis=0))[0]
    prediction_image = (prediction * 255).astype(np.uint8)  # Convert prediction to image format

    # img = cv2.imread(image_path)
    # resized_img = cv2.resize(img, (256, 256))
    # resized_img = resized_img/255

    ## Predict

    # prediction_image = model.predict(np.expand_dims(resized_img, axis=0))[0]


    # Correctly construct the path for the result image
    result_filename = str(os.path.basename(image_path))
    result_image_path = Path(settings.MEDIA_ROOT, "Result", result_filename)
    Image.fromarray(prediction_image).save(result_image_path)

    # Save the URL to Result.txt
    result_image_url = settings.MEDIA_URL + "Result/" + result_filename
    result_list_path = Path(settings.MEDIA_ROOT, "Result", "Result.txt")
    with result_list_path.open("a") as f:
        f.write(str(result_image_url) + "\n")
    print(result_image_path)
    return result_image_path


# Create your models here.
class ImagePage(Page):
    """Image Page."""

    template = "image_new.html"

    max_count = 2

    name_title = models.CharField(max_length=100, blank=True, null=True)
    name_subtitle = RichTextField(features=["bold", "italic"], blank=True)

    content_panels = Page.content_panels + [
        MultiFieldPanel(
            [
                FieldPanel("name_title"),
                FieldPanel("name_subtitle"),
            ],
            heading="Page Options",
        ),
    ]

    def reset_context(self, request):
        context = super().get_context(request)
        context["my_uploaded_file_names"] = []
        context["my_result_file_names"] = []
        context["my_processed_file_names"] = []  # Initialize the list here
        context["my_staticSet_names"] = []
        context["my_lines"]: []
        return context

    def serve(self, request):
        context = self.reset_context(request)
        emptyButtonFlag = False

        # Reset uploaded and processed image lists
        context["my_uploaded_file_names"] = []
        context["my_processed_file_names"] = []

        # Process images when the "Start" button is pressed
        if request.POST.get("start") == "":
            img_list_path = Path(settings.MEDIA_ROOT, "processed_images", "img_list.txt")
            result_list_path = Path(settings.MEDIA_ROOT, "Result", "Result.txt")
            uploaded_images_folder = Path(settings.MEDIA_ROOT, "uploadedPics")
            processed_images_folder = Path(settings.MEDIA_ROOT, "processed_images")
            print("Start processing images")
            if img_list_path.exists():
                with img_list_path.open("r") as file:
                    for line in file.readlines():
                        name = os.path.basename(line.strip())
                        image_path = Path(settings.MEDIA_ROOT, 'uploadedPics', name)
                        result_image_path = process_and_predict(str(image_path), unet_model)
                        result_image_url = Path(settings.MEDIA_URL, "Result", os.path.basename(result_image_path))
                        context["my_result_file_names"].append(result_image_url)
                        # Save the result URL to Result.txt
                        with result_list_path.open("a") as result_file:
                            result_file.write(str(result_image_url) + "\n")
                        # Retrieve raw uploaded images
                        print("uploaded_images_folder---------")            
                        for file_path in uploaded_images_folder.glob("*.jpg"):
                            context["my_uploaded_file_names"].append(settings.MEDIA_URL + "uploadedPics/" + file_path.name)
                            print(context["my_uploaded_file_names"])
                        # Retrieve processed images
                        print("processed_images_folder----ajkdaldkf")            
                        for file_path in processed_images_folder.glob("*.jpg"):
                            context["my_processed_file_names"].append(settings.MEDIA_URL + "processed_images/" + file_path.name)
                            print(context["my_processed_file_names"])

        # Process uploaded images when files are uploaded
        if request.FILES and emptyButtonFlag == False:
            reset()
            self.reset_context(request)
            # Path to save processed images
            processed_images_folder = Path(settings.MEDIA_ROOT, "processed_images")
            for file_obj in request.FILES.getlist("file_data"):
                uuidStr = uuid.uuid4()
                filename = f"{file_obj.name.split('.')[0]}_{uuidStr}.{file_obj.name.split('.')[-1]}"
                input_file_path = Path(settings.MEDIA_ROOT, 'uploadedPics', filename)
                output_file_path = Path(processed_images_folder, filename)
                # Save uploaded file
                with default_storage.open(input_file_path, "wb+") as destination:
                    for chunk in file_obj.chunks():
                        destination.write(chunk)
                # Call draw_random_lines function to process the image
                draw_random_lines(input_file_path, output_file_path)
                # Add the processed image to the context for rendering
                processed_image_url = Path(settings.MEDIA_URL, "processed_images", filename)
                context["my_processed_file_names"].append(processed_image_url)
                # Write the processed image URL to a text file or database if needed
                with open(Path(f"{settings.MEDIA_ROOT}/processed_images/img_list.txt"), "a") as f:
                    f.write(str(filename))
                    f.write("\n")
                # Add the uploaded image to the context for rendering
                uploaded_image_url = Path(settings.MEDIA_URL, "uploadedPics", filename)
                context["my_uploaded_file_names"].append(uploaded_image_url)

        # # Retrieve raw uploaded images
        # uploaded_images_folder = Path(settings.MEDIA_ROOT, "uploadedPics")
        # for file_path in uploaded_images_folder.glob("*.*"):
        #     context["my_uploaded_file_names"].append(settings.MEDIA_URL + "uploadedPics/" + file_path.name)
        #     print("Uploaded Images:")
        #     print(context["my_uploaded_file_names"])

        # # Retrieve processed images
        # processed_images_folder = Path(settings.MEDIA_ROOT, "processed_images")
        # for file_path in processed_images_folder.glob("*.*"):
        #     context["my_processed_file_names"].append(settings.MEDIA_URL + "processed_images/" + file_path.name)
        #     print("Processed Images:")
        #     print(context["my_processed_file_names"])

        print("This is final context---",context)
        return render(request, "image_new.html", context)