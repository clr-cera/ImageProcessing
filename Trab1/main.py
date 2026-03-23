import streamlit as st
import imageio.v3 as iio
from trans import Transformation
import intense
import numpy as np
st.title("Image Translation App")
st.text("Upload an image")

def on_image_upload():
    print("Image uploaded.")
    st.session_state["image"] = iio.imread(st.session_state["image_uploader"])
    st.session_state["canvas_size"] = st.session_state["image"].shape
    st.session_state["canvas_range"] = (0, st.session_state["canvas_size"][0], 0, st.session_state["canvas_size"][1])
st.file_uploader(label="Upload an image (png, jpg)", type=["png", "jpg"], key="image_uploader", on_change=on_image_upload)


if st.session_state.get("image", None) is not None:
    st.image(st.session_state["image"][st.session_state["canvas_range"][0]:st.session_state["canvas_range"][1], 
                                       st.session_state["canvas_range"][2]:st.session_state["canvas_range"][3]], 
                                       caption="Uploaded Image")
    print(f"Canvas range: {st.session_state['canvas_range']}")
    print(f"Image shape: {st.session_state['image'].shape}")

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
        
        # test bounds
        if st.session_state["canvas_range"][0] + dy < 0 or st.session_state["canvas_range"][1] + dy > st.session_state["image"].shape[0] or \
           st.session_state["canvas_range"][2] + dx < 0 or st.session_state["canvas_range"][3] + dx > st.session_state["image"].shape[1]:
            st.error("Translation would move the image outside the canvas.")
            should_apply = False

        if should_apply:
            st.session_state["canvas_range"] = (st.session_state["canvas_range"][0] + dy, st.session_state["canvas_range"][1] + dy, st.session_state["canvas_range"][2] + dx, st.session_state["canvas_range"][3] + dx)
            st.rerun(scope="app")
    
    st.subheader("Scale the image")
    scale_input = st.text_input("Enter sx value for scaling", key="sx_input")
    scale_input2 = st.text_input("Enter sy value for scaling", key="sy_input")
    if st.button("Apply Scaling"):
        should_apply = True
        try:
            sx = float(scale_input)
            sy = float(scale_input2)
        except ValueError:
            st.error("Please enter valid numbers for sx and sy.")
            should_apply = False
        
        if sx * st.session_state["image"].shape[1] < st.session_state["canvas_size"][1] or sy * st.session_state["image"].shape[0] < st.session_state["canvas_size"][0]:
            st.error("Scaling would make the image smaller than the canvas.")
            should_apply = False

        if should_apply:
            st.session_state["image"] = Transformation.scale_image(st.session_state["image"], sx, sy)
            # centralize canvas after image scaling
            st.session_state["canvas_range"] = (st.session_state["image"].shape[0]//2 - st.session_state["canvas_size"][0]//2, 
                                                st.session_state["image"].shape[0]//2 + st.session_state["canvas_size"][0]//2, 
                                                st.session_state["image"].shape[1]//2 - st.session_state["canvas_size"][1]//2, 
                                                st.session_state["image"].shape[1]//2 + st.session_state["canvas_size"][1]//2)
            st.rerun(scope="app")
    
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
            # Scale the image before rotation
            # According to someone on stackoverflow:
            # If w>h, the scaling factor is (w/h) sin(angle) + cos(angle).
            # If h>w, the scaling factor is (h/w) sin(angle) + cos(angle).
            current_scale = max(st.session_state["image"].shape[0] / st.session_state["canvas_size"][0], st.session_state["image"].shape[1] / st.session_state["canvas_size"][1])
            current_width, current_height = st.session_state["image"].shape[1], st.session_state["image"].shape[0]
            if current_width > current_height:
                scaling_factor = (current_width / current_height) * np.sin(np.radians(theta)) + np.cos(np.radians(theta))
            else:
                scaling_factor = (current_height / current_width) * np.sin(np.radians(theta)) + np.cos(np.radians(theta))
            final_scale = scaling_factor / current_scale
            if final_scale > 1:
                st.session_state["image"] = Transformation.scale_image(st.session_state["image"], final_scale, final_scale)
                st.session_state["canvas_range"] = (st.session_state["image"].shape[0]//2 - st.session_state["canvas_size"][0]//2, 
                                                    st.session_state["image"].shape[0]//2 + st.session_state["canvas_size"][0]//2, 
                                                    st.session_state["image"].shape[1]//2 - st.session_state["canvas_size"][1]//2, 
                                                    st.session_state["image"].shape[1]//2 + st.session_state["canvas_size"][1]//2)
            st.session_state["image"] = Transformation.rot_image(st.session_state["image"], theta)
            st.rerun(scope="app")
        
    
    st.subheader("Invert the image!")
    if st.button("Apply Inversion"):
        st.session_state["image"] = intense.inverse_intensity(st.session_state["image"])
        st.rerun(scope="app")

    st.subheader("Logarithmic transformation")
    if st.button("Apply Logarithmic Transformation"):
        st.session_state["image"] = intense.log(st.session_state["image"])
        st.rerun(scope="app")
    
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
            st.session_state["image"] = intense.gamma(st.session_state["image"], gamma_value)
            st.rerun(scope="app")

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
            st.session_state["image"] = intense.modulate_contrast(st.session_state["image"], contrast_limit)
            st.rerun(scope="app")

    st.subheader("Sigmoid transformation")
    if st.button("Apply Sigmoid Transformation"):
        st.session_state["image"] = intense.sigmoid(st.session_state["image"])
        st.rerun(scope="app")