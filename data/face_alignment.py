import dlib
from os.path import join
from PIL import Image


def make_align(face_file_path, target_folder, not_detected_file_path):

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_5_face_landmarks.dat")

    img = dlib.load_rgb_image(face_file_path)

    dets = detector(img, 1)

    num_faces = len(dets)

    if num_faces == 0:
        # print("Sorry, there were no faces found in '{}'".format(face_file_path))
        with open(not_detected_file_path, "a") as csv_file_name:
            csv_file_name.write(face_file_path + ",\n")
    else:
        # Find the 5 face landmarks we need to do the alignment.
        faces = dlib.full_object_detections()
        for detection in dets:
            faces.append(predictor(img, detection))

        # Getting Aligned and extracted images
        images = dlib.get_face_chips(img, faces, size=256)
        images = Image.fromarray(images[0])

        # Saving the cleaned images
        target_split = face_file_path.split("/")
        crnt_img = target_split[-1]
        images.save(join(target_folder, crnt_img), "JPEG")

