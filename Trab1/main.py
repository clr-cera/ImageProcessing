import streamlit as st
import imageio.v3 as iio
from trans import Transformation
import intense
from functools import partial

# Header
st.title("Image Transformation App")
st.text("Upload an image")

def on_image_upload():
    print("Image uploaded.")
    image = iio.imread(st.session_state["image_uploader"])
    # remove alpha channel if exists
    if image.shape[2] == 4:
        image = image[:, :, :3]
    st.session_state["image"] = image
    st.session_state["Transform"] = Transformation(st.session_state["image"].shape)
    st.session_state["IntensityPipeline"] = []

st.file_uploader(label="Upload an image (png, jpg)", type=["png", "jpg"], key="image_uploader", on_change=on_image_upload)


if st.session_state.get("image", None) is not None:
    # Display processed Image
    piped_image =  st.session_state["Transform"].transform_image(st.session_state["image"])
    for func in st.session_state["IntensityPipeline"]:
        piped_image = func(piped_image)

    st.image(piped_image, caption="Trans image")

    # Geometric transformations
    # Translation
    st.subheader("Trans the image")
    translation_input = st.text_input("Enter dx value for translation", key="dx_input")
    translation_input2 = st.text_input("Enter dy value for translation", key="dy_input")

    if st.button("Apply Translation"):
        should_apply = True
        try:
            dx = int(translation_input)
            dy = int(translation_input2)
        except ValueError:
            st.error("Please enter valid integer values for dx and dy.")
            should_apply = False

        # check corners
        if not st.session_state["Transform"].is_translation_valid(dx, dy):
            st.error("Translation would move part of the image out of bounds.")
            should_apply = False
        

        if should_apply:
            st.session_state["Transform"].translate(dx, dy)
            st.rerun(scope="app")
    
    # Scaling
    st.subheader("Scale the image")
    scale_input = st.text_input("Enter scale value for scaling", key="s_input")
    if st.button("Apply Scaling"):
        should_apply = True
        try:
            s = float(scale_input)
        except ValueError:
            st.error("Please enter valid numbers for scale.")
            should_apply = False

        if should_apply:
            st.session_state["Transform"].scale(s)
            st.rerun(scope="app")
    
    # Rotating
    st.subheader("Rotate the image")
    rotation_input = st.text_input("Enter theta value for rotation (degrees)", key="theta_input")
    if st.button("Apply Rotation"):
        should_apply = True
        try:
            theta = float(rotation_input)
        except ValueError:
            st.error("Please enter a valid number for theta.")
            should_apply = False

        if should_apply:
            st.session_state["Transform"].rotate(theta)
            st.rerun(scope="app")
        
    # Intensity transformations
    # Invert
    st.subheader("Invert the image!")
    if st.button("Apply Inversion"):
        st.session_state["IntensityPipeline"].append(intense.invert)
        st.rerun(scope="app")

    # Logarithmic
    st.subheader("Logarithmic transformation")
    if st.button("Apply Logarithmic Transformation"):
        st.session_state["IntensityPipeline"].append(intense.log)
        st.rerun(scope="app")
    
    # Gamma
    st.subheader("Gamma transformation")
    gamma_input = st.text_input("Enter gamma value for transformation", key="gamma_input")
    if st.button("Apply Gamma Transformation"):
        should_apply = True
        try:
            gamma_value = float(gamma_input)
        except ValueError:
            st.error("Please enter a valid number for gamma.")
            should_apply = False

        if should_apply:
            # Currying! :D
            st.session_state["IntensityPipeline"].append(partial(intense.gamma, gamma_value=gamma_value))
            st.rerun(scope="app")

    # Contrasting
    st.subheader("Modulate contrast")
    contrast_input = st.text_input("Enter contrast limit value for modulation", key="contrast_input")
    if st.button("Apply Contrast Modulation"):
        should_apply = True
        try:
            contrast_limit = float(contrast_input)
        except ValueError:
            st.error("Please enter a valid number for contrast limit.")
            should_apply = False

        if should_apply:
            # I love curry
            st.session_state["IntensityPipeline"].append(partial(intense.modulate_contrast, contrast_limit=contrast_limit))
            st.rerun(scope="app")

    # Sigmoiding (My function!)
    st.subheader("Sigmoid transformation")
    if st.button("Apply Sigmoid Transformation"):
        st.session_state["IntensityPipeline"].append(intense.sigmoid)
        st.rerun(scope="app")