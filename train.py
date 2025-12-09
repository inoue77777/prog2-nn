import time
import matplotlib.pyplot as plt
import torch
from torchvision import datasets
import torchvision.transforms.v2 as transforms
import models

# データセットの前処理を定義
ds_transform = transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True)
])


# データセットの読み込み
ds_train = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ds_transform
)
ds_test = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ds_transform
)

# ミニバッチを分割する。DataLoaderを作る。
batch_size = 64
dataloader_train = torch.utils.data.DataLoader(
    ds_train, batch_size=batch_size, shuffle=True
)
dataloader_test = torch.utils.data.DataLoader(
    ds_test, batch_size=batch_size
)

# モデルのインスタンスを作成
model = models.MyModel()

# 損失関数（誤差関数・ロス関数）の選択
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

n_epochs = 20

train_loss_log = []
val_loss_log = []
train_acc_log = []
val_acc_log = []

for epoch in range(n_epochs):
    print(f"epoch {epoch+1}/{n_epochs}")

    train_loss = models.train(model, dataloader_train, loss_fn, optimizer)
    print(f"  train loss: {train_loss:.4f}")
    train_loss_log.append(train_loss)

    val_loss = models.test(model, dataloader_test, loss_fn)
    print(f"  val loss: {val_loss:.4f}")
    val_loss_log.append(val_loss)

    train_acc = models.test_accuracy(model, dataloader_train)
    print(f"  train acc: {train_acc*100:.2f}%")
    train_acc_log.append(train_acc)

    val_acc = models.test_accuracy(model, dataloader_test)
    print(f"  val acc: {val_acc*100:.2f}%")
    val_acc_log.append(val_acc)


# グラフ描画
epochs = range(1, n_epochs + 1)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss_log, label='train')
plt.plot(epochs, val_loss_log, label='validation')
plt.xlabel("epochs")
plt.ylabel("loss")
plt.xticks(epochs)
plt.grid()
plt.legend()


plt.subplot(1, 2, 2)
plt.plot(epochs, train_acc_log, label='train')
plt.plot(epochs, val_acc_log, label='validation')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.xticks(epochs)
plt.grid()
plt.legend()

plt.show()
