__all__ = ["train_step", "val_step", "create_embedding"]

import torch
import torch.nn as nn

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_step(encoder, decoder, train_loader, loss_fn, optimizer, device):
    """
    执行一个训练步骤
    
    参数:
    - encoder: 卷积编码器。例如：torch_model ConvEncoder
    - decoder: 卷积解码器。例如：torch_model ConvDecoder
    - train_loader: PyTorch dataloader, 包含 (images, images).
    - loss_fn: PyTorch loss_fn, 计算两幅图像之间的损失（差值，loss）
    - optimizer: PyTorch optimizer.
    - device: "cuda" 或者 "cpu"

    返回值: 训练损失
    """
    encoder.train()
    decoder.train()

    # print(device)

    for batch_idx, (train_img, target_img) in enumerate(train_loader):
        train_img = train_img.to(device)
        target_img = target_img.to(device)

        optimizer.zero_grad()

        enc_output = encoder(train_img)
        dec_output = decoder(enc_output)

        loss = loss_fn(dec_output, target_img)
        loss.backward()

        optimizer.step()

    return loss.item()


def val_step(encoder, decoder, val_loader, loss_fn, device):
    """
    执行一个训练步骤
    
    参数:
    - encoder: 卷积编码器。例如：torch_model ConvEncoder
    - decoder: 卷积解码器。例如：torch_model ConvDecoder
    - val_loader: PyTorch dataloader, 包含 (images, images).
    - loss_fn: PyTorch loss_fn, 计算两幅图像之间的损失（差值，loss）
    - device: "cuda" 或者 "cpu"

    返回值: 验证损失
    """

    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        for batch_idx, (train_img, target_img) in enumerate(val_loader):
            train_img = train_img.to(device)
            target_img = target_img.to(device)

            enc_output = encoder(train_img)
            dec_output = decoder(enc_output)

            loss = loss_fn(dec_output, target_img)

    return loss.item()


def create_embedding(encoder, full_loader, embedding_dim, device):
    """
    使用编码器为数据创建嵌入
    
    参数：
    - encoder: 卷积编码器。例如：torch_model ConvEncoder
    - full_loader: PyTorch dataloader, 包含整个数据集的 (images, images) 
    - embedding_dim: Tuple (c, h, w) 嵌入维度 = 编码器输出的维度。
    - device: "cuda" 或者 "cpu"

    返回值: 嵌入的大小 (num_images_in_loader + 1, c, h, w)
    """
    encoder.eval()
    embedding = torch.randn(embedding_dim)
    print("embedding.shape: ", embedding.shape)

    with torch.no_grad():
        for batch_idx, (train_img, target_img) in enumerate(full_loader):
            train_img = train_img.to(device)
            enc_output = encoder(train_img).cpu()
            embedding = torch.cat((embedding, enc_output), 0)
            # print(embedding.shape)

    return embedding
