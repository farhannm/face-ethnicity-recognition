import cv2
import numpy as np
from mtcnn import MTCNN
import tensorflow as tf

class MTCNNFaceDetector:
    """
    Face detector based on Multi-task Cascaded Convolutional Networks (MTCNN)
    """
    def __init__(self, min_face_size=20, factor=0.709):
        """
        Initialize the MTCNN face detector
        
        Args:
            min_face_size: Minimum face size to detect (default: 20)
            factor: Scale factor for the image pyramid (default: 0.709)
        """
        # Hide TensorFlow warnings about CPU/GPU
        tf.get_logger().setLevel('ERROR')
        
        self.detector = MTCNN(
            min_face_size=min_face_size,
            scale_factor=factor
        )
        self.target_size = (224, 224)  # Default target size for face images
        
    def detect_faces(self, image):
        """
        Detect faces in the input image
        
        Args:
            image: Input image (BGR format from OpenCV)
            
        Returns:
            face_boxes: List of tuples (x, y, w, h) for each detected face
            face_images: List of cropped face images
            landmarks: List of facial landmarks for each detected face
        """
        # Convert BGR to RGB (MTCNN expects RGB)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        detections = self.detector.detect_faces(rgb_image)
        
        face_boxes = []
        face_images = []
        landmarks = []
        
        for detection in detections:
            # Get bounding box
            x, y, w, h = detection['box']
            
            # Add margin to face (20%)
            margin_x = int(w * 0.2)
            margin_y = int(h * 0.2)
            
            # Make sure coordinates are within image boundaries
            x1 = max(0, x - margin_x)
            y1 = max(0, y - margin_y)
            x2 = min(image.shape[1], x + w + margin_x)
            y2 = min(image.shape[0], y + h + margin_y)
            
            # Convert back to (x, y, w, h) format
            face_box = (x1, y1, x2 - x1, y2 - y1)
            face_boxes.append(face_box)
            
            # Crop face region
            face_img = image[y1:y2, x1:x2]
            
            # Resize to target size
            if face_img.size > 0:  # Check if face_img is not empty
                face_img = cv2.resize(face_img, self.target_size)
                face_images.append(face_img)
            
            # Get landmarks
            landmarks.append({
                'left_eye': detection['keypoints']['left_eye'],
                'right_eye': detection['keypoints']['right_eye'],
                'nose': detection['keypoints']['nose'],
                'mouth_left': detection['keypoints']['mouth_left'],
                'mouth_right': detection['keypoints']['mouth_right']
            })
        
        return face_boxes, face_images, landmarks
    
    def detect_and_align_face(self, image, target_size=(224, 224)):
        """
        Detect faces and perform alignment based on eye positions
        
        Args:
            image: Input image (BGR format from OpenCV)
            target_size: Target size for the aligned face
            
        Returns:
            aligned_face: Aligned face image (or None if no face detected)
            face_found: Boolean indicating whether a face was found
            face_rect: Face rectangle coordinates (x, y, w, h) or None
            confidence: Detection confidence score or None
        """
        # Convert BGR to RGB (MTCNN expects RGB)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        detections = self.detector.detect_faces(rgb_image)
        
        # If no face detected, return resized original image
        if len(detections) == 0:
            resized = cv2.resize(image, target_size)
            return resized, False, None, None
        
        # Get the detection with highest confidence
        detection = max(detections, key=lambda x: x['confidence'])
        confidence = detection['confidence']
        
        # Get bounding box
        x, y, w, h = detection['box']
        face_rect = (x, y, w, h)
        
        # Add margin to face (20%)
        margin_x = int(w * 0.2)
        margin_y = int(h * 0.2)
        
        # Make sure coordinates are within image boundaries
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(image.shape[1], x + w + margin_x)
        y2 = min(image.shape[0], y + h + margin_y)
        
        # Crop face region
        face_img = image[y1:y2, x1:x2]
        
        # Get eye coordinates (within the face image coordinate system)
        left_eye = (detection['keypoints']['left_eye'][0] - x1, 
                    detection['keypoints']['left_eye'][1] - y1)
        right_eye = (detection['keypoints']['right_eye'][0] - x1, 
                     detection['keypoints']['right_eye'][1] - y1)
        
        # Calculate angle for alignment
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        
        if dx > 0:
            angle = np.degrees(np.arctan2(dy, dx))
            
            # Rotation center between eyes
            center_x = (left_eye[0] + right_eye[0]) // 2
            center_y = (left_eye[1] + right_eye[1]) // 2
            center = (center_x, center_y)
            
            # Rotation matrix
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            # Apply rotation
            aligned_face = cv2.warpAffine(
                face_img, 
                rotation_matrix, 
                (face_img.shape[1], face_img.shape[0]),
                flags=cv2.INTER_CUBIC, 
                borderMode=cv2.BORDER_CONSTANT
            )
        else:
            aligned_face = face_img
            
        # Resize to target size
        normalized_face = cv2.resize(aligned_face, target_size)
        
        # Apply contrast normalization using CLAHE
        if len(normalized_face.shape) == 3:
            # For color images, normalize in LAB color space
            lab = cv2.cvtColor(normalized_face, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            
            normalized_lab = cv2.merge((cl, a, b))
            normalized_face = cv2.cvtColor(normalized_lab, cv2.COLOR_LAB2BGR)
        else:
            # For grayscale images
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            normalized_face = clahe.apply(normalized_face)
            
        return normalized_face, True, face_rect, confidence
    
    def draw_faces(self, image, faces, landmarks=None, color=(0, 255, 0), thickness=2):
        """
        Draw bounding boxes and landmarks around detected faces
        
        Args:
            image: Input image
            faces: List of face coordinates (x, y, w, h)
            landmarks: List of facial landmarks dictionaries
            color: Color of the bounding box (BGR format)
            thickness: Line thickness
            
        Returns:
            image_with_faces: Image with drawn face bounding boxes and landmarks
        """
        image_with_faces = image.copy()
        
        for i, (x, y, w, h) in enumerate(faces):
            # Draw face rectangle
            cv2.rectangle(image_with_faces, (x, y), (x+w, y+h), color, thickness)
            
            # Draw landmarks if available
            if landmarks and i < len(landmarks):
                landmark = landmarks[i]
                
                # Draw landmarks as circles
                for point_name, point in landmark.items():
                    cv2.circle(image_with_faces, point, 2, (0, 0, 255), -1)
            
        return image_with_faces