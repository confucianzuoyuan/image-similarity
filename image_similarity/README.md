# How the code is structured ?

- `torch_data.py` 包含了 dataset 类，用来加载数据。
- `torch_model.py` 包含了我们要训练的模型。
- `torch_engine.py` 包含了训练步骤和验证步骤，还包含了创建嵌入的代码。
- `utils.py` 包含了一些功能函数。
- `torch_train.py` 包含了训练脚本。用到了 `torch_data.py`, `torch_model.py`, `torch_engine.py` 和 `utils.py` 。
- `torch_infer.py` 包含了推理代码。

- `.ipynb` notebooks 提供了单独可运行的python脚本，用来训练模型和推理。

[dataset link](https://www.kaggle.com/datasets/okeaditya/animals-data)

- `web_app.py` 编写了一个 http server 来托管图片搜索服务。