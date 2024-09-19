import os
import face_recognition
import numpy as np


def generate_face_encodings(images_folder='Images', encodings_folder='Encodings'):
    # Create the encodings folder if it doesn't exist
    if not os.path.exists(encodings_folder):
        os.makedirs(encodings_folder)

    # Iterate over each subfolder (each person)
    for person in os.listdir(images_folder):
        person_folder = os.path.join(images_folder, person)
        encodings = []

        # Check if the folder is a directory
        if os.path.isdir(person_folder):
            # Iterate over each image in the person's folder
            for image_name in os.listdir(person_folder):
                image_path = os.path.join(person_folder, image_name)

                # Load the image and find face encodings
                image = face_recognition.load_image_file(image_path)
                face_locations = face_recognition.face_locations(image)
                face_encodings = face_recognition.face_encodings(image, face_locations)

                # Save the first detected face encoding, if available
                if face_encodings:
                    encodings.append(face_encodings[0])

            # Calculate the average encoding for the person
            if encodings:
                avg_encoding = np.mean(encodings, axis=0)
                encoding_path = os.path.join(encodings_folder, f'{person}.npy')
                np.save(encoding_path, avg_encoding)
                print(f"Encodings saved for {person}.")


# Call the function to generate encodings
generate_face_encodings()