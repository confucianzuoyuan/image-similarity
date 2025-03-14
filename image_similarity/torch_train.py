# 自编码器的训练脚本。

import torch
import torch_model
import torch_engine
import torchvision.transforms as T
import torch_data
import config
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import utils

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print("设置训练模型的随机数种子, seed = {}".format(config.SEED))

    utils.seed_everything(config.SEED)

    transforms = T.Compose([T.Resize((64, 64)), T.ToTensor()])
    print("------------ 创建数据集 ------------")
    full_dataset = torch_data.FolderDataset(config.IMG_PATH, transforms)

    train_size = int(config.TRAIN_RATIO * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )

    print("------------ 数据集创建完成 ------------")
    print("------------ 创建数据加载器 ------------")
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.TRAIN_BATCH_SIZE, shuffle=True, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config.TEST_BATCH_SIZE
    )
    full_loader = torch.utils.data.DataLoader(
        full_dataset, batch_size=config.FULL_BATCH_SIZE
    )

    print("------------ 数据加载器创建完成 ------------")

    # print(train_loader)
    loss_fn = nn.MSELoss()

    encoder = torch_model.ConvEncoder()
    decoder = torch_model.ConvDecoder()

    if torch.cuda.is_available():
        print("GPU 可用，可以将模型移动到 GPU。")
    else:
        print("可以将模型移动到CPU。")

    encoder.to(device)
    decoder.to(device)

    # print(device)

    autoencoder_params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.AdamW(autoencoder_params, lr=config.LEARNING_RATE)

    # early_stopper = utils.EarlyStopping(patience=5, verbose=True, path=)
    max_loss = 9999

    print("------------ Training started ------------")

    for epoch in tqdm(range(config.EPOCHS)):
        train_loss = torch_engine.train_step(
            encoder, decoder, train_loader, loss_fn, optimizer, device=device
        )
        print(f"Epochs = {epoch}, Training Loss : {train_loss}")
        val_loss = torch_engine.val_step(
            encoder, decoder, val_loader, loss_fn, device=device
        )

        # 保存最好的模型
        if val_loss < max_loss:
            print("验证集的损失减小了，保存新的最好的模型。")
            torch.save(encoder.state_dict(), config.ENCODER_MODEL_PATH)
            torch.save(decoder.state_dict(), config.DECODER_MODEL_PATH)

        print(f"Epochs = {epoch}, Validation Loss : {val_loss}")

    print("训练结束")

    print("---- 对整个数据集创建嵌入 ---- ")

    embedding = torch_engine.create_embedding(
        encoder, full_loader, config.EMBEDDING_SHAPE, device
    )

    # 将嵌入转换成numpy数据类型，并保存
    numpy_embedding = embedding.cpu().detach().numpy()
    num_images = numpy_embedding.shape[0]

    # 将嵌入转成可以保存的类型
    flattened_embedding = numpy_embedding.reshape((num_images, -1))
    np.save(config.EMBEDDING_PATH, flattened_embedding)
