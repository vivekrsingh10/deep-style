# app.py - The Streamlit Web App for DeepStyle

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time

# --- Copied and Refactored from your original deepstyle.py ---

# We cache the model loading function to avoid reloading it on every run.
# This makes the app much faster.
@st.cache_resource
def load_vgg_model(style_layers, content_layers):
    """ Creates a VGG model that returns a list of intermediate output values."""
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    outputs = [vgg.get_layer(name).output for name in (style_layers + content_layers)]
    model = tf.keras.Model([vgg.input], outputs)
    return model

class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg = load_vgg_model(style_layers, content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        inputs = inputs * 255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                          outputs[self.num_style_layers:])
        style_outputs = [gram_matrix(style_output) for style_output in style_outputs]
        content_dict = {name: value for name, value in zip(self.content_layers, content_outputs)}
        style_dict = {name: value for name, value in zip(self.style_layers, style_outputs)}
        return {'content': content_dict, 'style': style_dict}

def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result/(num_locations)

def load_img(image_bytes):
    """Loads an image from bytes, limits its maximum dimension, and converts to a tensor."""
    max_dim = 512
    img = tf.image.decode_image(image_bytes, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim
    new_shape = tf.cast(shape * scale, tf.int32)
    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img

def run_style_transfer(content_image, style_image, epochs=10, steps_per_epoch=100):
    """The main function to run the style transfer."""
    content_layers = ['block5_conv2']
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
    
    extractor = StyleContentModel(style_layers, content_layers)
    style_targets = extractor(style_image)['style']
    content_targets = extractor(content_image)['content']

    generated_image = tf.Variable(content_image)
    optimizer = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

    content_weight = 1e4
    style_weight = 1e-2

    # This is the optimization loop
    for n in range(epochs):
        for m in range(steps_per_epoch):
            with tf.GradientTape() as tape:
                outputs = extractor(generated_image)
                style_outputs = outputs['style']
                content_outputs = outputs['content']
                style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2) for name in style_outputs.keys()])
                style_loss *= style_weight / len(style_layers)
                content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2) for name in content_outputs.keys()])
                content_loss *= content_weight / len(content_layers)
                loss = style_loss + content_loss
            
            grad = tape.gradient(loss, generated_image)
            optimizer.apply_gradients([(grad, generated_image)])
            generated_image.assign(tf.clip_by_value(generated_image, clip_value_min=0.0, clip_value_max=1.0))
        st.write(f"Epoch {n+1} completed.") # Provide feedback during the long process

    return generated_image

# --- The Streamlit App UI ---

st.title("🎨 DeepStyle Web App")
st.write("Upload a content image and a style image to create a new, stylized masterpiece!")

# Create two columns for the file uploaders
col1, col2 = st.columns(2)

with col1:
    st.header("Content Image")
    content_file = st.file_uploader("Choose a content image...", type=["jpg", "jpeg", "png"])

with col2:
    st.header("Style Image")
    style_file = st.file_uploader("Choose a style image...", type=["jpg", "jpeg", "png"])

# Check if both files have been uploaded
if content_file and style_file:
    # Display the uploaded images
    col1.image(content_file, caption='Your Content Image', use_container_width=True)
    col2.image(style_file, caption='Your Style Image', use_container_width=True)

    # A button to start the process
    if st.button("Stylize My Image!"):
        # Show a spinner while the model runs
        with st.spinner('Creating your masterpiece... This can take a few minutes.'):
            # Load images from the uploaded file bytes
            content_image_bytes = content_file.getvalue()
            style_image_bytes = style_file.getvalue()
            
            content_tensor = load_img(content_image_bytes)
            style_tensor = load_img(style_image_bytes)

            # Run the model
            final_image = run_style_transfer(content_tensor, style_tensor, epochs=10, steps_per_epoch=100) # Reduced for faster web demo

            # Squeeze the batch dimension and convert to a displayable format
            final_image_to_display = np.squeeze(final_image.numpy())

        # Display the final image
        st.header("Your DeepStyle Result")
        st.image(final_image_to_display, caption='Stylized Image', use_container_width=True)

else:
    st.warning("Please upload both a content and a style image to begin.")