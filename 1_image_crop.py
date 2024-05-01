import cv2
import dlib
import os

def crop_face(image_path, output_path, padding_factor=0.9):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image at {image_path}")
        return

    # Initialize face detector
    detector = dlib.get_frontal_face_detector()

    # Detect faces in the image
    faces = detector(image, 1)
    
    if len(faces) > 0:
        # For the purposes of this example, we'll use the first detected face
        face = faces[0]
        
        # Get the size of the face
        face_width = face.right() - face.left()
        face_height = face.bottom() - face.top()
        
        # Calculate the size of the square crop we're aiming for
        side_length = max(face_width, face_height)
        
        # Introduce padding
        padding = side_length * padding_factor  # padding relative to the face size
        
        # Calculate the center of the face
        center_x = face.left() + face_width // 2
        center_y = face.top() + face_height // 2
        
        # Calculate the points for the square crop including padding
        x1 = max(center_x - int(side_length // 2 + padding), 0)
        y1 = max(center_y - int(side_length // 2 + padding), 0)
        x2 = min(center_x + int(side_length // 2 + padding), image.shape[1])
        y2 = min(center_y + int(side_length // 2 + padding), image.shape[0])
        
        # Ensure the crop is still a square if it reaches the image boundary
        square_side = min(x2 - x1, y2 - y1)
        x2 = x1 + square_side
        y2 = y1 + square_side

        # Crop the image
        cropped_image = image[y1:y2, x1:x2]
        
        # Save the cropped image
        cv2.imwrite(output_path, cropped_image)
    else:
        print("No face detected in the image.")

if __name__ == '__main__':
    base_input_folder = './input/images/'
    base_output_folder = './input/cropped/'

    prompts = os.listdir(base_input_folder)
    print(f'Prompts: {prompts}')

    for prompt in prompts:
        input_folder = os.path.join(base_input_folder, prompt)
        output_folder = os.path.join(base_output_folder, prompt)

        # Check if the directory exists and is indeed a directory
        if os.path.isdir(input_folder):
            # Create output folder if not exists
            os.makedirs(output_folder, exist_ok=True)

            list_of_images = os.listdir(input_folder)
            list_of_images = [img for img in list_of_images if img.endswith('.jpg')]
            list_of_images = sorted(list_of_images)

            for image_name in list_of_images:
                output_image_path = os.path.join(output_folder, image_name)

                if os.path.exists(output_image_path):
                    print(f"Skipping image: {image_name}")
                    continue

                print(f"Processing image: {image_name}")
                input_image_path = os.path.join(input_folder, image_name)
                # Call the crop_face function to process the image
                crop_face(input_image_path, output_image_path)