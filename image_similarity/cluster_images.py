##### 使用 PCA 和 T-SNE 对嵌入进行聚类 #########

__all__ = ["cluster_images", "vizualise_tsne"]

import numpy as np
import pickle
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import config
import matplotlib.pyplot as plt


def cluster_images(embedding, pca_num_components: int, tsne_num_components: int):
    """
    使用 PCA + T-SNE 算法对图片进行聚类，以及可视化 T-SNE
    
    参数:
    - embedding: 图像嵌入的 2 维向量表示
    - pca_num_components: Number of componenets PCA should reduce.
    - tsne_num_components: Number of componenets T-SNE should reduce to. Suggested: 2
    """
    pca_file_name = f"..//data//models//pca_{pca_num_components}.pkl"
    tsne_file_name = f"..//data//models//tsne_{tsne_num_components}.pkl"
    tsne_embeddings_file_name = (
        f"..//data//models//tsne_embeddings_{tsne_num_components}.pkl"
    )

    print("使用 PCA 算法降维")

    pca = PCA(n_components=pca_num_components, random_state=42)
    reduced_embedding = pca.fit_transform(embedding)
    # print(reduced_embedding.shape)

    # 使用 T-SNE 聚类
    print("使用 T-SNE 聚类")
    tsne_obj = TSNE(
        n_components=tsne_num_components,
        verbose=1,
        random_state=42,
        perplexity=200,
        n_iter=1000,
        n_jobs=-1,
    )

    tsne_embedding = tsne_obj.fit_transform(reduced_embedding)

    # print(tsne_embedding.shape)

    # 保存 TSNE 和 PCA 对象.
    pickle.dump(pca, open(pca_file_name, "wb"))
    # pickle.dump(tsne_embedding)
    pickle.dump(tsne_obj, open(tsne_file_name, "wb"))

    # 可视化 TSNE.
    vizualise_tsne(tsne_embedding)

    # 保存嵌入.
    pickle.dump(tsne_embedding, open(tsne_embeddings_file_name, "wb"))


def vizualise_tsne(tsne_embedding):
    """
    可视化 T-SNE 嵌入
    
    参数:
    tsne_embedding: 2 维 T-SNE 嵌入
    """

    x = tsne_embedding[:, 0]
    y = tsne_embedding[:, 1]

    plt.scatter(x, y, c=y)
    plt.show()


if __name__ == "__main__":
    # 加载嵌入
    embedding = np.load(config.EMBEDDING_PATH)

    # print(embedding.shape)

    # Reshape Back to Encoder Embeddings.
    # NUM_IMAGES = (4739, )
    # embedding_shape = NUM_IMAGES + config.EMBEDDING_SHAPE[1:]
    # print(embedding_shape)

    # embedding = np.reshape(embedding, embedding_shape)
    # print(embedding.shape)

    cluster_images(embedding, pca_num_components=50, tsne_num_components=2)
