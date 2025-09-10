import cv2
from openface.face_detection import FaceDetector

# Initialize the FaceDetector
model_path = './OpenFace-3.0/weights/Alignment_RetinaFace.pth'
detector = FaceDetector(model_path=model_path, device='cuda')

# Path to the input image
image_path = 'path/to/input_image.jpg'

# Detect and extract the face
cropped_face, dets = detector.get_face(image_path)

if cropped_face is not None:
    print("Face detected!")
    print(f"Detection results: {dets}")
    
    # Save the cropped face as an image
    output_path = 'path/to/output_face.jpg'
    cv2.imwrite(output_path, cropped_face)
    print(f"Detected face saved to: {output_path}")
else:
    print("No face detected.")