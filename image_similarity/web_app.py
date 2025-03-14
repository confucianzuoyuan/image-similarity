# Let's make the app

from flask import Flask, request, json
import torch_model
import config
import torch
import numpy as np

from sklearn.neighbors import NearestNeighbors
import torchvision.transforms as T
import os
import cv2
from PIL import Image
from sklearn.decomposition import PCA

app = Flask(__name__)

print("启动应用")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# 在启动服务器之前加载模型
encoder = torch_model.ConvEncoder()
# 加载 encoder 的 state dict
encoder.load_state_dict(torch.load(config.ENCODER_MODEL_PATH, map_location=device))
encoder.eval()
encoder.to(device)
# 加载嵌入
embedding = np.load(config.EMBEDDING_PATH)

print("模型和嵌入已加载完成")


def compute_similar_images(image_tensor, num_images, embedding, device):
    """
    给定一张图像和要生成的相似图像的数量。
    返回 num_images 张最相似的图像的数量

    参数:
    - image_tenosr: 通过 PIL 将图像转换成的张量 image_tensor ，需要寻找和 image_tensor 相似的图像。
    - num_images: 要寻找的相似图像的数量。
    - embedding : 一个 (num_images, embedding_dim) 元组，是从自编码器学到的图像的嵌入。
    - device : "cuda" 或者 "cpu" 设备。
    """

    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        image_embedding = encoder(image_tensor).cpu().detach().numpy()

    # print(image_embedding.shape)

    flattened_embedding = image_embedding.reshape((image_embedding.shape[0], -1))
    # print(flattened_embedding.shape)

    # 使用 KNN 算法寻找最近邻的图像
    knn = NearestNeighbors(n_neighbors=num_images, metric="cosine")
    knn.fit(embedding)

    _, indices = knn.kneighbors(flattened_embedding)
    indices_list = indices.tolist()
    # print(indices_list)
    return indices_list


def compute_similar_features(image, num_images, embedding, nfeatures=30):
    """
    给定一张图像，使用 ORB detector 计算特征并查找具有相似特征的图像。
    
    参数:
    - image: 通过 Opencv 读取的 Image 类型的图像，查找和 image 的特征最接近的图像。
    - num_images: 需要查找的相似图像的数量。
    - embedding: 2 维嵌入向量。
    - nfeatures: (可选) ORB 需要计算的特征数量。
    """

    orb = cv2.ORB_create(nfeatures=nfeatures)

    # 探测特征
    keypoint_features = orb.detect(image)
    # 使用 ORB 计算特征
    keypoint_features, des = orb.compute(image, keypoint_features)

    # des 包含特征的描述

    des = des / 255.0
    des = np.expand_dims(des, axis=0)
    des = np.reshape(des, (des.shape[0], -1))
    # print(des.shape)
    # print(embedding.shape)

    pca = PCA(n_components=des.shape[-1])
    reduced_embedding = pca.fit_transform(
        embedding,
    )
    # print(reduced_embedding.shape)

    knn = NearestNeighbors(n_neighbors=num_images, metric="cosine")
    knn.fit(reduced_embedding)
    _, indices = knn.kneighbors(des)

    indices_list = indices.tolist()
    # print(indices_list)
    return indices_list


# 首页
@app.route("/")
def index():
    return "应用启动成功"


@app.route("/simfeat", methods=["POST"])
def simfeat():
    r = request.files["image"]
    # 将字符串格式的图像数据转换成uint8类型
    nparr = np.fromstring(r.data, np.uint8)
    # 对图像进行解码
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    indices_list = compute_similar_features(img, num_images=5, embedding=embedding)
    # 返回给前端显示图像
    return (
        json.dumps({"indices_list": indices_list}),
        200,
        {"ContentType": "application/json"},
    )


@app.route("/simimages", methods=["POST"])
def simimages():
    # 从请求中获取图像数据
    image = request.files["image"]
    # 作为图像打开
    image = Image.open(image)
    # 转换成张量
    image_tensor = T.ToTensor()(image)
    # 增加 1 个维度
    image_tensor = image_tensor.unsqueeze(0)
    # 计算并返回相似的图像
    indices_list = compute_similar_images(
        image_tensor, num_images=5, embedding=embedding, device=device
    )
    # 返回给前端显示图像
    return (
        json.dumps({"indices_list": indices_list}),
        200,
        {"ContentType": "application/json"},
    )


if __name__ == "__main__":
    app.run(debug=False, port=9000)
