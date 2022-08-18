import face_cut

FILEDIR = "../Data/"
train_image, train_labels = face_cut.cut_image(FILEDIR)

print(train_image, train_labels)