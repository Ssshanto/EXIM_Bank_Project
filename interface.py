import streamlit as st
import os
import json
from PIL import Image
import io
import numpy as np
from nid_recog import get_nid_info
from face_recog import detect_face

# Configure page
st.set_page_config(
    page_title="EXIM Bank",
    layout="wide"
)

# Main title
st.title("NID & Face Recognition")
st.markdown("---")

# Create tabs
tab1, tab2 = st.tabs(["NID Information", "Face Recognition"])

# NID Information Tab
with tab1:
    st.header("NID Information Extraction")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an NID card image...", 
        type=['jpg', 'jpeg', 'png', 'webp', 'bmp', 'tiff', 'tif', 'gif'],
        help="Supported formats: JPG, JPEG, PNG, WebP, BMP, TIFF, GIF"
    )
    
    if uploaded_file is not None:
        # Display the uploaded image
        st.subheader("Uploaded Image")
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded NID Card", use_container_width=True)
        
        # Process button
        if st.button("Extract NID Information", type="primary"):
            with st.spinner("Processing NID"):
                try:
                    # Save uploaded file temporarily
                    temp_path = f"temp_{uploaded_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Process the image
                    nid_info = get_nid_info(temp_path)
                    
                    # Clean up temporary file
                    os.remove(temp_path)
                    
                    # Display results
                    st.success("NID Information extracted successfully!")
                    
                    # Create a nice display of the results
                    st.subheader("Extracted Information")
                    
                    # Create columns for better layout
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Personal Information:**")
                        if nid_info.get('Name'):
                            st.write(f"**Name:** {nid_info['Name']}")
                        if nid_info.get('Name_Bangla'):
                            st.write(f"**Name (Bangla):** {nid_info['Name_Bangla']}")
                        if nid_info.get('Date_of_Birth'):
                            st.write(f"**Date of Birth:** {nid_info['Date_of_Birth']}")
                        if nid_info.get('NID_Number'):
                            st.write(f"**NID Number:** {nid_info['NID_Number']}")
                    
                    with col2:
                        st.markdown("**Family Information:**")
                        if nid_info.get("Father's Name"):
                            st.write(f"**Father's Name:** {nid_info['Father\'s Name']}")
                        if nid_info.get("Mother's Name"):
                            st.write(f"**Mother's Name:** {nid_info['Mother\'s Name']}")
                    
                    # Show raw JSON data in expander
                    with st.expander("Raw JSON Data"):
                        st.json(nid_info)
                    
                    # Download button for results
                    json_str = json.dumps(nid_info, indent=2, ensure_ascii=False)
                    st.download_button(
                        label="Download Results as JSON",
                        data=json_str,
                        file_name=f"nid_info_{uploaded_file.name.split('.')[0]}.json",
                        mime="application/json"
                    )
                    
                except FileNotFoundError as e:
                    st.error(f"File not found: {e}")
                except ValueError as e:
                    st.error(f"Invalid input: {e}")
                except Exception as e:
                    st.error(f"Error processing image: {e}")
                    st.info("Make sure your API key is properly configured in the .env file.")

# Face Recognition Tab
with tab2:
    st.header("Face Recognition")
    
    # Two side-by-side file uploaders
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_file1 = st.file_uploader(
            "Upload first image...",
            type=['jpg', 'jpeg', 'png', 'webp', 'bmp', 'tiff', 'tif', 'gif'],
            key="file1"
        )
    
    with col2:
        uploaded_file2 = st.file_uploader(
            "Upload second image...",
            type=['jpg', 'jpeg', 'png', 'webp', 'bmp', 'tiff', 'tif', 'gif'],
            key="file2"
        )
    
    # Display uploaded images if both are provided
    if uploaded_file1 is not None and uploaded_file2 is not None:
        st.subheader("Uploaded Images")
        
        # Display images side by side
        img_col1, img_col2 = st.columns(2)
        
        with img_col1:
            image1 = Image.open(uploaded_file1)
            st.image(image1, caption="Image 1", use_container_width=True)
        
        with img_col2:
            image2 = Image.open(uploaded_file2)
            st.image(image2, caption="Image 2", use_container_width=True)
        
        # Process button
        if st.button("Compare Faces", type="primary"):
            with st.spinner("Processing face comparison"):
                try:
                    # Save uploaded files temporarily
                    temp_path1 = f"temp_face1_{uploaded_file1.name}"
                    temp_path2 = f"temp_face2_{uploaded_file2.name}"
                    
                    with open(temp_path1, "wb") as f:
                        f.write(uploaded_file1.getbuffer())
                    with open(temp_path2, "wb") as f:
                        f.write(uploaded_file2.getbuffer())
                    
                    # Process the images
                    results = detect_face(temp_path1, temp_path2)
                    
                    # Clean up temporary files
                    os.remove(temp_path1)
                    os.remove(temp_path2)
                    
                    # Display results
                    st.success("Face comparison completed!")
                    
                    # Display images with cropped faces
                    st.subheader("Face Analysis")
                    
                    # Create 2x2 grid for images
                    img_row1_col1, img_row1_col2 = st.columns(2)
                    img_row2_col1, img_row2_col2 = st.columns(2)
                    
                    with img_row1_col1:
                        st.markdown("**Original Image 1**")
                        st.image(image1, use_container_width=True)
                    
                    with img_row1_col2:
                        st.markdown("**Original Image 2**")
                        st.image(image2, use_container_width=True)
                    
                    with img_row2_col1:
                        st.markdown("**Cropped Face 1**")
                        if results['face1_cropped'] is not None:
                            # Convert numpy array to PIL Image
                            face1_pil = Image.fromarray(results['face1_cropped'])
                            st.image(face1_pil, use_container_width=True)
                        else:
                            st.error("No face detected in Image 1")
                    
                    with img_row2_col2:
                        st.markdown("**Cropped Face 2**")
                        if results['face2_cropped'] is not None:
                            # Convert numpy array to PIL Image
                            face2_pil = Image.fromarray(results['face2_cropped'])
                            st.image(face2_pil, use_container_width=True)
                        else:
                            st.error("No face detected in Image 2")
                    
                    # Display statistics
                    st.subheader("Comparison Results")
                    
                    # Create columns for statistics
                    stat_col1, stat_col2, stat_col3 = st.columns(3)
                    
                    with stat_col1:
                        st.metric("Verification Result", 
                                "Match" if results['verified'] else "No Match",
                                delta="✓" if results['verified'] else "✗")
                    
                    with stat_col2:
                        st.metric("Similarity Score", 
                                f"{results['distance']:.4f}",
                                delta="Lower is better")
                    
                    with stat_col3:
                        st.metric("Threshold", 
                                f"{results['threshold']:.4f}")
                    
                    # Display detailed statistics
                    st.subheader("Detailed Statistics")
                    
                    detail_col1, detail_col2 = st.columns(2)
                    
                    with detail_col1:
                        st.markdown("**Face Detection:**")
                        st.write(f"**Image 1 faces detected:** {results['faces_detected_img1']}")
                        st.write(f"**Image 2 faces detected:** {results['faces_detected_img2']}")
                        st.write(f"**Model used:** {results['model_name']}")
                    
                    with detail_col2:
                        st.markdown("**Processing Details:**")
                        st.write(f"**Processing time:** {results['processing_time']:.2f} seconds")
                        st.write(f"**Distance metric:** {results['distance_metric']}")
                        st.write(f"**Verification threshold:** {results['threshold']}")
                    
                    # Show raw results in expander
                    with st.expander("Raw Results Data"):
                        # Remove numpy arrays from results for JSON serialization
                        json_results = {k: v for k, v in results.items() if not isinstance(v, np.ndarray)}
                        st.json(json_results)
                    
                    # Download button for results
                    json_str = json.dumps(json_results, indent=2, ensure_ascii=False)
                    st.download_button(
                        label="Download Results as JSON",
                        data=json_str,
                        file_name=f"face_comparison_{uploaded_file1.name.split('.')[0]}_{uploaded_file2.name.split('.')[0]}.json",
                        mime="application/json"
                    )
                    
                except FileNotFoundError as e:
                    st.error(f"File not found: {e}")
                except ValueError as e:
                    st.error(f"Invalid input: {e}")
                except Exception as e:
                    st.error(f"Error processing images: {e}")
                    st.info("Make sure both images contain clear, visible faces.")

# Sidebar with API status only
with st.sidebar:
    st.header("API Status")
    try:
        from nid_recog import get_api_key
        api_key = get_api_key()
        st.success("Gemini API Key: Configured")
        st.caption(f"Using: {api_key[:10]}...{api_key[-4:]}")
    except Exception as e:
        st.error("Gemini API Key: Not configured")
        st.caption("Please set up your API key in the .env file")
