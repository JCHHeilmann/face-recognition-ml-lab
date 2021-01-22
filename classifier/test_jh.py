from PIL import Image
from data.face_alignment import FaceAlignment

img = Image.open("classifier/PeopleKnown/Alex Rodriguez/AlexRodriguez0.jpg").convert(
    "RGB"
)

face_alignment = FaceAlignment()

aligned_image = face_alignment.make_align(img)

aligned_image.show()
