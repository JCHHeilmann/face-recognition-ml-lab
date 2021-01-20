import os
import shutil
import dlib
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from classifier.l2distanze_classifier import L2DistanzClassifier


Classifier = L2DistanzClassifier()

if os.path.exists("./PeopleKnown/Lewandowski"):
    shutil.rmtree("./PeopleKnown/Lewandowski", ignore_errors=True)

result = Classifier.classify("./PeopleUnknown/Lewandowski_Test.jpeg")
print(result)

Classifier.add_person("./PeopleUnknown/Lewandowski_1.jpeg", "Lewandowski")
Classifier.add_person("./PeopleUnknown/Lewandowski_2.jpeg", "Lewandowski")
Classifier.add_person("./PeopleUnknown/Lewandowski_3.jpeg", "Lewandowski")
Classifier.add_person("./PeopleUnknown/Lewandowski_4.jpeg", "Lewandowski")
Classifier.add_person("./PeopleUnknown/Lewandowski_5.jpeg", "Lewandowski")
Classifier.add_person("./PeopleUnknown/Lewandowski_6.jpeg", "Lewandowski")

result2 = Classifier.classify("./PeopleUnknown/Lewandowski_Test.jpeg")
print(result2)
