import streamlit as st
from transformers import YolosImageProcessor, YolosForObjectDetection
from PIL import Image
import torch
import requests

st.title("Object Detection with YOLOS")

# User input for image URL
url = st.text_input("Enter image URL:")
if url:
    try:
        image = Image.open(requests.get(url, stream=True).raw)
    except:
        st.error("Invalid image URL or unable to load image. Please try again.")
        st.stop()
else:
    st.warning("Please enter an image URL.")

# Load YOLOS model
model = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny')
image_processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")

# Perform object detection if image is loaded successfully
if url:
    # Perform object detection
    inputs = image_processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # Model predicts bounding boxes and corresponding COCO classes
    logits = outputs.logits
    bboxes = outputs.pred_boxes

    # Post-process and display results
    target_sizes = torch.tensor([image.size[::-1]])
    results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[0]

    st.subheader("Object Detection Results")
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        result_str = f"Detected {model.config.id2label[label.item()]} with confidence {round(score.item(), 3)} at location {box}"
        st.write(result_str)

    # Display the input image
    st.subheader("Input Image")
    st.image(image, caption="Input Image", use_column_width=True)
