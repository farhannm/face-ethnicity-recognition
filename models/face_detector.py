import cv2
import numpy as np

class HaarCascadeFaceDetector:
    """
    Face detector based on Haar Cascade Classifier from OpenCV
    """
    def __init__(self, cascade_path=None):
        """
        Initialize the face detector
        
        Args:
            cascade_path: Path to cascade classifier XML file. If None, uses default frontal face classifier.
        """
        if cascade_path is None:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        else:
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            
        self.target_size = (224, 224)  # Default target size for face images
        
    def detect_faces(self, image, min_neighbors=5, scale_factor=1.1, min_size=(30, 30)):
        """
        Detect faces in the input image
        
        Args:
            image: Input image (BGR format from OpenCV)
            min_neighbors: Minimum neighbors parameter for the classifier
            scale_factor: Scale factor for the detection
            min_size: Minimum size of the face to be detected
            
        Returns:
            faces: List of tuples (x, y, w, h) for each detected face
            face_images: List of cropped face images
        """
        # Convert to grayscale for detection
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=scale_factor, 
            minNeighbors=min_neighbors,
            minSize=min_size
        )
        
        # Extract face images
        face_images = []
        for (x, y, w, h) in faces:
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
            
            # Resize to target size
            if face_img.size > 0:  # Check if face_img is not empty
                face_img = cv2.resize(face_img, self.target_size)
                face_images.append(face_img)
        
        return faces, face_images
    
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
        """
        # Convert to grayscale for detection
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # If no face detected, return resized original image
        if len(faces) == 0:
            resized = cv2.resize(image, target_size)
            return resized, False, None
        
        # Get the largest face
        if len(faces) > 1:
            face_areas = [w*h for (x, y, w, h) in faces]
            largest_face_idx = np.argmax(face_areas)
            (x, y, w, h) = faces[largest_face_idx]
        else:
            (x, y, w, h) = faces[0]
            
        face_rect = (x, y, w, h)
        
        # Extract face region with margin
        margin_x = int(w * 0.2)
        margin_y = int(h * 0.2)
        
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(image.shape[1], x + w + margin_x)
        y2 = min(image.shape[0], y + h + margin_y)
        
        face_img = image[y1:y2, x1:x2]
        
        # Extract face region for eye detection
        face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY) if len(face_img.shape) == 3 else face_img.copy()
        
        # Detect eyes within the face region
        eyes = self.eye_cascade.detectMultiScale(face_gray, 1.1, 3, minSize=(20, 20))
        
        # If less than 2 eyes detected, skip alignment
        if len(eyes) < 2:
            normalized_face = cv2.resize(face_img, target_size)
            return normalized_face, True, face_rect
        
        # Get eye centers
        eye_centers = [(ex + ew//2, ey + eh//2) for ex, ey, ew, eh in eyes]
        
        # Find all possible eye pairs
        eye_pairs = [(i, j) for i in range(len(eye_centers)) for j in range(i+1, len(eye_centers))]
        
        # Choose the pair with largest horizontal distance
        if eye_pairs:
            best_pair = max(eye_pairs, key=lambda pair: abs(eye_centers[pair[0]][0] - eye_centers[pair[1]][0]))
            left_eye, right_eye = eye_centers[best_pair[0]], eye_centers[best_pair[1]]
            
            # Make sure left eye is on the left
            if left_eye[0] > right_eye[0]:
                left_eye, right_eye = right_eye, left_eye
                
            # Calculate rotation angle
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
            
        return normalized_face, True, face_rect
    
    def draw_faces(self, image, faces, color=(0, 255, 0), thickness=2):
        """
        Draw bounding boxes around detected faces
        
        Args:
            image: Input image
            faces: List of face coordinates (x, y, w, h)
            color: Color of the bounding box (BGR format)
            thickness: Line thickness
            
        Returns:
            image_with_faces: Image with drawn face bounding boxes
        """
        image_with_faces = image.copy()
        
        for (x, y, w, h) in faces:
            cv2.rectangle(image_with_faces, (x, y), (x+w, y+h), color, thickness)
            
        return image_with_faces