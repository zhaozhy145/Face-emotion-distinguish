import util
import cnn_emotion_distinguish
TEST_DIR_PATH = '../Data/fer2013_data_strength/test'
test_data, test_labels = util.read_images_list(TEST_DIR_PATH)
cnn_emotion_distinguish.T(test_data, test_labels)