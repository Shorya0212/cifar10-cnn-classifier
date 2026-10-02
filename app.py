import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import cv2
import numpy as np
from model import CNN



st.set_page_config(
    page_title="CIFAR-10 AI Classifier",
    page_icon="🤖",
    layout="wide"
)



st.markdown("""
# 🤖 CIFAR-10 Image Classifier
Upload an image or choose a demo image and the CNN model will classify it into one of the **10 CIFAR-10 categories**.
""")



classes = [
'airplane','automobile','bird','cat','deer',
'dog','frog','horse','ship','truck'
]


model = CNN()
model.load_state_dict(torch.load("cifar10_cnn.pth", map_location="cpu"))
model.eval()



transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])



def generate_gradcam(model, img_tensor):

    gradients = []
    activations = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    target_layer = model.conv_layers[-3]

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_backward_hook(backward_hook)

    output = model(img_tensor)
    pred_class = output.argmax()

    model.zero_grad()
    output[0, pred_class].backward()

    grads = gradients[0].cpu().data.numpy()[0]
    acts = activations[0].cpu().data.numpy()[0]

    weights = np.mean(grads, axis=(1,2))

    cam = np.zeros(acts.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = np.maximum(cam,0)
    cam = cam / cam.max()

    forward_handle.remove()
    backward_handle.remove()

    return cam




col1, col2 = st.columns([1,1])



with col1:

    st.subheader("Input Image")

    uploaded_file = st.file_uploader(
        "Upload your own image",
        type=["jpg","jpeg","png","webp"]
    )

    if uploaded_file:

        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

    else:

        st.subheader("Choose a demo image")

        sample = st.selectbox(
            "Select a class",
            classes
        )

        image = Image.open(f"images/{sample}.webp")
        st.image(image, caption="Demo Image", use_container_width=True)




with col2:

    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img)
        probabilities = F.softmax(outputs, dim=1)[0]
        confidence, predicted = torch.max(probabilities,0)

    st.subheader("Prediction")

    st.success(f"{classes[predicted.item()].upper()}")

    st.subheader("Prediction Confidence")

    probs = probabilities.numpy()

    for i in range(len(classes)):
        st.progress(float(probs[i]), text=f"{classes[i]} : {probs[i]*100:.2f}%")



st.divider()

cam = generate_gradcam(model, img)

cam = cv2.resize(cam, (image.size[0], image.size[1]))

heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

original = np.array(image)

overlay = heatmap * 0.4 + original

st.subheader("Model Attention (Grad-CAM)")

st.image(overlay.astype(np.uint8), use_container_width=True)