# ==============================================================================
# SECTION 1: IMPORTS AND SETUP
# ==============================================================================
# We're importing the libraries we installed for the DeepStyle project.
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time

print(f"TensorFlow Version: {tf.__version__}")
print("DeepStyle Project Initializing...")

# ==============================================================================
# SECTION 2: IMAGE LOADING AND PREPROCESSING
# ==============================================================================
# This function will load our images and make sure they are the right size.
def load_img(path_to_img):
    """Loads an image, limits its maximum dimension to 512 pixels, and prepares it."""
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :] # Add a batch dimension
    return img

# This function is a helper to display the images.
def imshow(image, title=None):
    """Displays a tensor as an image."""
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)

    plt.imshow(image)
    if title:
        plt.title(title)
    plt.axis('off')

# ==============================================================================
# SECTION 3: DEFINE THE MODEL AND LOSS FUNCTIONS
# ==============================================================================
# Define which layers from the VGG19 model we'll use for content and style.
content_layers = ['block5_conv2']
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

def vgg_layers(layer_names):
    """
    Creates the VGG19 model. This pre-trained model acts as our "art expert",
    extracting features from the images.
    """
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False

    outputs = [vgg.get_layer(name).output for name in layer_names]
    model = tf.keras.Model([vgg.input], outputs)
    return model

def gram_matrix(input_tensor):
    """
    The Gram matrix is the key to capturing style. It measures the co-occurrence
    of features, which corresponds to textures and patterns in the image.
    """
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result/(num_locations)

class StyleContentModel(tf.keras.models.Model):
    """The main model that extracts style and content features from an image."""
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg = vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        "Expects float input in range [0,1]"
        inputs = inputs * 255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                          outputs[self.num_style_layers:])

        style_outputs = [gram_matrix(style_output) for style_output in style_outputs]

        content_dict = {name: value for name, value in zip(self.content_layers, content_outputs)}
        style_dict = {name: value for name, value in zip(self.style_layers, style_outputs)}

        return {'content': content_dict, 'style': style_dict}

# These weights determine the trade-off between content and style.
content_weight = 1e4
style_weight = 1e-2

def style_content_loss(outputs, style_targets, content_targets):
    """Calculates the total loss, which we will try to minimize."""
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - style_targets[name])**2)
                           for name in style_outputs.keys()])
    style_loss *= style_weight / num_style_layers

    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - content_targets[name])**2)
                             for name in content_outputs.keys()])
    content_loss *= content_weight / num_content_layers
    loss = style_loss + content_loss
    return loss

# ==============================================================================
# SECTION 4: THE TRAINING STEP
# ==============================================================================
@tf.function()
def train_step(image, extractor, style_targets, content_targets, optimizer):
    """
    Performs one optimization step to update the generated image.
    """
    with tf.GradientTape() as tape:
        outputs = extractor(image)
        loss = style_content_loss(outputs, style_targets, content_targets)

    grad = tape.gradient(loss, image)
    optimizer.apply_gradients([(grad, image)])
    image.assign(tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0))

# ==============================================================================
# SECTION 5: RUN THE DEEPSTYLE PROCESS
# ==============================================================================
if __name__ == '__main__':
    # DEFINE IMAGE PATHS
    content_path = 'images/content.jpg'
    style_path = 'images/style.jpg'

    # LOAD AND DISPLAY IMAGES
    print("Loading content and style images...")
    content_image = load_img(content_path)
    style_image = load_img(style_path)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    imshow(content_image, 'Content Image')
    plt.subplot(1, 2, 2)
    imshow(style_image, 'Style Image')
    plt.suptitle("DeepStyle Inputs")
    plt.show()

    # CREATE THE MAIN EXTRACTOR MODEL
    extractor = StyleContentModel(style_layers, content_layers)

    # PRE-CALCULATE THE TARGET STYLE AND CONTENT FEATURES
    print("Extracting target features from style and content images...")
    style_targets = extractor(style_image)['style']
    content_targets = extractor(content_image)['content']

    # INITIALIZE THE IMAGE TO BE GENERATED (starting with the content image)
    generated_image = tf.Variable(content_image)

    # CREATE THE OPTIMIZER
    optimizer = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

    # --- THE MAIN OPTIMIZATION LOOP ---
    print("Starting style transfer. This will take several minutes...")
    start_time = time.time()

    epochs = 10
    steps_per_epoch = 100

    for n in range(epochs):
        for m in range(steps_per_epoch):
            train_step(generated_image, extractor, style_targets, content_targets, optimizer)
        print(f"Epoch {n+1} of {epochs} completed.")

    end_time = time.time()
    print(f"Total time: {end_time - start_time:.1f} seconds")

    # --- DISPLAY AND SAVE THE FINAL IMAGE ---
    print("DeepStyle process finished. Displaying final result.")
    plt.figure(figsize=(10, 10))
    imshow(generated_image, f"DeepStyle Result (Epochs: {epochs})")
    plt.show()

    # Save the generated image
    final_image_tensor = tf.squeeze(generated_image, axis=0)
    final_image_tensor = tf.image.convert_image_dtype(final_image_tensor, tf.uint8)
    pil_image = Image.fromarray(final_image_tensor.numpy())
    pil_image.save('deepstyle_output.jpg')
    print("Saved final image as 'deepstyle_output.jpg'")