import dlib
from os.path import join
from PIL import Image


def make_align(face_file_path, target_folder, not_detected_file_path):

    detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")
    predictor = dlib.shape_predictor("shape_predictor_5_face_landmarks.dat")

    img = dlib.load_rgb_image(face_file_path)

    detections = [det.rect for det in detector(img, 1)]

    num_faces = len(detections)

    if num_faces == 0:
        # print("Sorry, there were no faces found in '{}'".format(face_file_path))
        with open(not_detected_file_path, "a") as csv_file_name:
            csv_file_name.write(face_file_path + ",\n")
    else:
        # Find the 5 face landmarks we need to do the alignment.
        face = dlib.full_object_detections()

        face.append(predictor(img, detections[0]))  # only use one of the faces

        # Getting Aligned and extracted images
        image = dlib.get_face_chips(img, face, size=256)
        output_image = Image.fromarray(image[0])

        # Saving the cleaned images
        target_split = face_file_path.split("/")
        crnt_img = target_split[-1]
        output_image.save(join(target_folder, crnt_img), "JPEG")

