# Face Detection and Comparison System

This project provides a comprehensive face detection and comparison system using DeepFace's FaceNet512 model. The system can verify whether two faces belong to the same person, extract facial landmarks, and provide detailed analysis with visualizations.

## Features

- **Face Verification**: Uses DeepFace's FaceNet512 model to verify if two faces belong to the same person
- **Landmark Extraction**: Extracts and visualizes facial landmarks (eyes, nose, mouth, eyebrows)
- **Face Cropping**: Automatically crops and aligns face regions
- **Embedding Analysis**: Compares face embedding vectors for similarity
- **Comprehensive Visualization**: Creates detailed plots showing:
  - Original images
  - Cropped face regions
  - Facial landmarks
  - Embedding vector comparisons
  - Similarity scores
- **Detailed Statistics**: Provides comprehensive analysis including:
  - Verification results
  - Similarity scores
  - Landmark counts
  - Embedding statistics
  - Interpretation guidelines

## Installation

1. Clone this repository or download the files
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from face_recog import detect_face

# Compare two face images
results = detect_face("path/to/image1.jpg", "path/to/image2.jpg")
```

### Function Parameters

- `img1_path` (str): Path to the first image file
- `img2_path` (str): Path to the second image file

### Function Returns

The function returns a dictionary containing:

- `verification_result`: DeepFace verification results
- `similarity_score`: Distance score between faces
- `face1_landmarks` / `face2_landmarks`: Facial landmark coordinates
- `face1_embedding` / `face2_embedding`: Face embedding vectors
- `face1_cropped` / `face2_cropped`: Cropped face regions
- `face1_original` / `face2_original`: Original images

## Output

The function provides:

1. **Console Output**: Step-by-step progress and detailed statistics
2. **Visualization**: 8-panel plot showing all analysis results
3. **Return Value**: Dictionary with all analysis data

### Console Output Example

```
============================================================
FACE DETECTION AND COMPARISON ANALYSIS
============================================================
Image 1: path/to/image1.jpg
Image 2: path/to/image2.jpg

1. PERFORMING FACE VERIFICATION...
   Verification Result: SAME PERSON
   Similarity Score: 0.2345
   Threshold: 0.3000

2. EXTRACTING FACE EMBEDDINGS AND LANDMARKS...
   Face 1 - Detected: 1 face(s)
   Face 2 - Detected: 1 face(s)

3. EXTRACTING CROPPED FACE REGIONS...
   Face regions extracted successfully

4. CREATING VISUALIZATION...

5. DETAILED STATISTICS:
   ==================================================
   VERIFICATION RESULTS:
   ==================================================
   Result: ✓ SAME PERSON
   Distance Score: 0.234500
   Threshold: 0.300000
   Model Used: FaceNet512
   Detector: OpenCV
```

## Visualization Panels

The function creates a comprehensive visualization with 8 panels:

1. **Original Image 1**: Full original image
2. **Original Image 2**: Full original image  
3. **Cropped Face 1**: Extracted face region
4. **Cropped Face 2**: Extracted face region
5. **Face 1 Landmarks**: Facial landmarks overlaid on image
6. **Face 2 Landmarks**: Facial landmarks overlaid on image
7. **Embedding Comparison**: First 50 dimensions of face embeddings
8. **Similarity Score**: Bar chart with threshold line

## Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff)

## Model Information

- **Face Detection**: OpenCV Haar Cascade
- **Face Recognition**: FaceNet512 (512-dimensional embeddings)
- **Landmark Detection**: DeepFace's built-in landmark extractor

## Similarity Score Interpretation

- **< 0.3**: High similarity - Very likely the same person
- **0.3 - 0.5**: Moderate similarity - Possibly the same person
- **> 0.5**: Low similarity - Likely different people

## Error Handling

The function includes comprehensive error handling for:

- Missing image files
- No faces detected in images
- Multiple faces in images
- Invalid image formats
- DeepFace processing errors

## Dependencies

- `deepface`: Face recognition and analysis
- `opencv-python`: Image processing and face detection
- `numpy`: Numerical computations
- `matplotlib`: Visualization
- `tensorflow`: Deep learning backend
- `retina-face`: Alternative face detector
- `mtcnn`: Multi-task Cascaded Convolutional Networks

## Example

```python
# Example usage
try:
    results = detect_face("person1_photo1.jpg", "person1_photo2.jpg")
    print("Analysis completed successfully!")
except Exception as e:
    print(f"Error: {e}")
```

## Notes

- The first time you run the function, DeepFace will download the required models automatically
- Processing time depends on image size and system performance
- Ensure good lighting and clear face visibility for best results
- The function works best with front-facing faces

## License

This project is open source and available under the MIT License.