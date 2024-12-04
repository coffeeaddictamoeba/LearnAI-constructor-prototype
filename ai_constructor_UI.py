# to run the code, enter "streamlit run path/to/your/python_file" to console
# do not forget to exit by ctrl c

import streamlit as st
import numpy as np

from PIL import Image
from almus.train import CNN
from dataset import LearnAIDataset

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" # temporary solution for conflicting libraries, needs to be fixed

# # For faster NN training use GPU
# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)

# TODO: OPTIMIZE IT.

# Models and presets
MODELS = ['Multilayer Perceptron', 'Coming Soon']
MODEL_PATHS = ['models/cats_and_dogs.h5', 'models/real_and_fake.h5'] # you can change it to your own path for trained models storage
PRESETS = ['Multilayer Perceptron', 'Coming Soon']
BATCH_SIZE = 32

# Datasets
# paths can be changed, but the train and test data should be separated to different files. The example of data organization can be seen in utils.dataset.py
DATASETS = [
    LearnAIDataset('almus/data/cats_and_dogs_train',
                   'almus/data/cats_and_dogs_test',
                   image_size=(200, 200),
                   num_classes=2,
                   dataset_name='Cats & Dogs'),
    LearnAIDataset('almus/data/fake_and_real_faces_train',
                   'almus/data/fake_and_real_faces_test',
                   image_size=(300, 300),
                   num_classes=2,
                   dataset_name='Real & AI Generated Person'),
    LearnAIDataset('no_path', 'no path',(1, 1), 4, 'Seasons of Year')
]

def get_dataset_by_name(name):
    for dataset in DATASETS:
        if dataset.dataset_name == name:
            return dataset
    return None

def extract_class_name(file_path):
    dir_name = os.path.dirname(file_path)
    class_name = os.path.basename(dir_name)
    return class_name

# Custom CSS for centering elements and customizing slider/button layout
st.markdown("""
    <style>
        /* Center-align the page title */
        .title {
            text-align: center;
            font-size: 36px;
            font-weight: bold;
        }
        /* Center-align buttons and sliders */
        .center-align {
            display: flex;
            justify-content: center;
            align-items: center;
            text-align: center;
        }
        .slider {
            width: 50%;  /* Adjust the width as needed */
            margin: 10px auto;
        }
        .button {
            display: block;
            margin: 20px auto;
        }
    </style>
""", unsafe_allow_html=True)

# Page Title
st.markdown('<div class="title">AI Constructor</div>', unsafe_allow_html=True)

# Sidebar logic (dataset, model, presets)
st.sidebar.title("Construct your own AI!")
st.sidebar.write("Here you can try out several of the most popular AI models, and you can set the parameters for training them yourself - just choose the necessary blocks!")

# Dataset choice
st.sidebar.header("Datasets")
dataset_name = st.sidebar.selectbox("Choose any dataset you like!", options=[ds.dataset_name for ds in DATASETS])
dataset = get_dataset_by_name(dataset_name)

# Get categories from chosen dataset
categories = dataset.get_categories()
print(f"Categories in the dataset: {categories}") # please ensure the categories order matches the categories order in train and test sets

# Fetch sample images for preview
sample_images = dataset.get_sample_images(num_samples=3)
st.write(f"### Preview of the {dataset.dataset_name} Dataset")
columns = st.columns(3)
for idx, image_path in enumerate(sample_images):
    col = columns[idx % 3]  # Cycle through columns
    with col:
        img = Image.open(image_path).resize(dataset.image_size)
        st.image(img, use_column_width=True)

# TODO: write a feature to load your own dataset (not for MVP)
st.sidebar.button("Add my own dataset!", key='my_ds', use_container_width=True)

# Model choice
st.sidebar.header("AI Models")
model_type = st.sidebar.selectbox("What type of AI model do you want to train?", options=MODELS)
user_model = None
hidden_layers = 0

# Preset choice (not for MVP)
st.sidebar.header("Presets")
preset_model = st.sidebar.selectbox("Check out the ready-made models!", options=PRESETS)

# Main Page Content
st.write(f"### Let's build your **{model_type}** model!")


# NEURAL NETWORK LOGIC
if model_type == 'Multilayer Perceptron':
    # Model initialization
    user_model = CNN((dataset.image_size[0], dataset.image_size[0], 3), num_classes=dataset.num_classes)

    # States to control the page view
    mlp_states = ["add_input_layer_clicked",
                  "first_layer_configured",
                  "hidden_layers_configured",
                  "skip_button_clicked",
                  "add_output_layer_clicked",
                  "output_layer_configured",
                  "training_end",
                  "history"]

    for state in mlp_states:
        if state not in st.session_state:
            st.session_state[state] = False

    # default values for NN
    st.session_state['input_layer_neurons'] = dataset.image_size[0]
    st.session_state['hidden_layer_neurons'] = 256
    st.session_state['activation_function_hidden'] = 'relu'
    st.session_state['dropout'] = 0.5
    st.session_state['activation_function_output'] = 'softmax'


    with st.container():
        st.markdown('<div class="center-align">', unsafe_allow_html=True)

        # Input Layer Configuration
        if st.button("Add Input Layer", use_container_width=True, key="input_layer_button"):
            st.session_state["add_input_layer_clicked"] = True
        if st.session_state.add_input_layer_clicked:
            input_layer_neurons = st.slider(
                label=f"The default value is the **size of the image** in the chosen dataset. Usually, it helps to reach the maximal performance, so you do not need to change anything. The amount of current input layer neurons is **{dataset.image_size[0]}**",
                min_value=1,
                max_value=dataset.image_size[0],
                value=st.session_state.input_layer_neurons,
                disabled=True
            )

            if st.button("To the next step!"):
                st.session_state["first_layer_configured"] = True
                st.success(f"Input Layer with {input_layer_neurons} neurons added!")
        st.markdown('</div>', unsafe_allow_html=True)

        # Hidden Layers or Skip (user choice)
        if st.session_state["first_layer_configured"]:
            st.write(f"Input Layer with {input_layer_neurons} neurons is configured.")

            with st.container():
                st.markdown('<div class="center-align">', unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Adjust Hidden Layers", use_container_width=True, key="add_hidden_layers_button"):
                        st.session_state["hidden_layers_configured"] = True
                with col2:
                    if st.button("Skip", use_container_width=True, key="skip_button"):
                        st.session_state["skip_button_clicked"] = True
                        st.write("The hidden layers will be configured with their default values.")
                st.markdown('</div>', unsafe_allow_html=True)

        # Hidden Layer settings
        if st.session_state.hidden_layers_configured:
            allowed_values = [2 ** i for i in range(5, 9)]  # [32, 64, 128, 256]

            hidden_layer_neurons = st.select_slider(
                label="Number of neurons for the first hidden layer:",
                options=allowed_values,
                value=256,  # Default value
                help="Usually the number of neurons in the hidden layers are some power of 2, for example, 32 or 256."
            )
            st.session_state["hidden_layer_neurons"] = hidden_layer_neurons

            activation_function_hidden = st.selectbox(
                label="Choose the Activation Function of your hidden layer:",
                options=['relu', 'sigmoid', 'tanh']
            )
            st.session_state["activation_function_hidden"] = activation_function_hidden

            dropout = st.slider(
                label="For better *generalization performance* (ability of neural network to adapt to new data), choose the dropout value:",
                min_value=0.1,
                max_value=0.5,
                value=0.5,
                step=0.05
            )
            st.session_state['dropout'] = dropout

            if st.button("Confirm Hidden Layer", key="hidden_button"):
                st.success(f"Hidden Layers with {hidden_layer_neurons} neurons and {activation_function_hidden} activation function added!")

        # Output Layer
        if st.session_state["hidden_layers_configured"] or st.session_state["skip_button_clicked"]:
            with st.container():
                st.markdown('<div class="center-align">', unsafe_allow_html=True)

                if st.button("Add Output Layer", use_container_width=True, key="output_layer_button"):
                    st.session_state["add_output_layer_clicked"] = True

                if st.session_state.add_output_layer_clicked:
                    output_layer_neurons = st.slider(
                        label=f"*The number of neurons for the output layer is usually equal to the number of classes in classification problems, so you do not need to change it. Your number of classes is {dataset.num_classes}",
                        min_value=1,
                        max_value=dataset.num_classes * 2,
                        value=dataset.num_classes,
                        step=1,
                        key="output_neurons",
                        disabled=True
                    )

                    activation_function_output = st.selectbox(
                        label="Choose the Activation Function of your output layer (for better performance, **do not choose the same function**, as in your hidden layer. Default choice for hidden layer is ReLU):",
                        options=['softmax', 'relu', 'sigmoid', 'tanh']
                    )
                    st.session_state["activation_function_output"] = activation_function_output

                    if st.button("Finish Configuration"):
                        st.session_state["output_layer_configured"] = True
                st.markdown('</div>', unsafe_allow_html=True)

    # Neural Network Configuration & Training
    if st.session_state["output_layer_configured"]:
        # Epochs Input
        epochs = st.number_input(
            "Well done! Your multilayer Perceptron is ready for training! Set the number of epochs for training your neural network:",
            min_value=1,
            max_value=10,
            value=3,
            step=1,
            help="As Multilayer Perceptron is very accurate, it tends to *overfit*, or learn the training dataset by heart *without noticing the pattern*. To avoid this, use the smaller number of epochs."
        )

        # Button to Train the Model
        if st.button("Train my Neural Network!"):
            with st.spinner("Training the neural network... Please wait."):
                st.write("*Training can take some time. Please be patient and wait for result.")

                # # Model Training Logic. Uncomment this if you want to test real-time training or models for every dataset is not ready
                # xception = CNN(input_shape=(input_layer_neurons, input_layer_neurons, 3), num_classes=dataset.num_classes)
                #
                # model = xception.build_model(
                #     activation_hidden=st.session_state["activation_function_hidden"],
                #     activation_output=st.session_state["activation_function_output"],
                #     dropout=st.session_state["dropout"]
                # )
                # xception.compile_model()
                #
                # history = xception.train(
                #     train_generator=dataset.augment_data_for_training(BATCH_SIZE),
                #     test_generator=dataset.augment_data_for_testing(BATCH_SIZE),
                #     epochs=epochs,
                #     filepath=MODEL_PATHS[DATASETS.index(dataset)]) # path to model should match the dataset name

                st.session_state["training_end"] = True

    cur_model = None
    test_gen=None
    if st.session_state["training_end"]:
        with st.spinner("Evaluating your neural network... Please wait."):

            cur_model = CNN(input_shape=(dataset.image_size[0], dataset.image_size[0], 3), model=MODEL_PATHS[DATASETS.index(dataset)]) # 3 for RGB

            # Evaluate the model
            test_gen = dataset.augment_data_for_testing(BATCH_SIZE)
            evaluation_results = cur_model.evaluate(test_generator=test_gen)
            st.success("Training complete!")
            st.write(f"Accuracy: {evaluation_results[1]:.2f}")
            st.write(f"Loss: {evaluation_results[0]:.2f}")
            st.session_state.history = True

    # Test Gallery
    if st.session_state["history"]:
        st.write("### Test Your Model with our Sample Images")
        num_samples = st.slider(
            "Choose the number of test images to display",
            min_value=3,
            max_value=9,
            value=5
        )

        # Displaying Test Images
        sample_images = dataset.get_sample_images(num_samples=num_samples, resize_to=dataset.image_size)
        for img_path, resized_image in sample_images:  # Unpack path and image tuple
            with st.container():

                # Display image
                st.image(resized_image, width=dataset.image_size[0])

                # Preprocess image for model prediction
                image_data = np.expand_dims(np.array(resized_image) / 255.0, axis=0)  # Normalize pixel values
                prediction = cur_model.predict(image_data)
                class_indices = {v: k for k, v in test_gen.class_indices.items()}
                predicted_label = class_indices[np.argmax(prediction)]

                # Display prediction
                st.write(f"Predicted Label: **{predicted_label}** - Real Label: **{extract_class_name(img_path)}**")

    # TODO: add logic for loading users own images for prediction. The possible way of realization can be found in "predict.py" file.