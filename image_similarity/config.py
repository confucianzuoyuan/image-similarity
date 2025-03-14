IMG_PATH = "../input/animals-data/dataset/"
IMG_HEIGHT = 512  # 图片的高度
IMG_WIDTH = 512  # 图片的宽度

SEED = 42
TRAIN_RATIO = 0.75
VAL_RATIO = 1 - TRAIN_RATIO
SHUFFLE_BUFFER_SIZE = 100

LEARNING_RATE = 1e-3
EPOCHS = 30
TRAIN_BATCH_SIZE = 32
TEST_BATCH_SIZE = 32
FULL_BATCH_SIZE = 32

###### 模型和嵌入相关 #########

DATA_PATH = "../data/images/"
AUTOENCODER_MODEL_PATH = "baseline_autoencoder.pt"
ENCODER_MODEL_PATH = "../data/models/deep_encoder.pt"
DECODER_MODEL_PATH = "../data/models/deep_decoder.pt"
EMBEDDING_PATH = "../data/models/data_embedding_f.npy"
EMBEDDING_SHAPE = (1, 256, 16, 16)
# TEST_RATIO = 0.2

###### 测试数据路径 #########
NUM_IMAGES = 10
TEST_IMAGE_PATH = "../data/images/2485.jpg"
