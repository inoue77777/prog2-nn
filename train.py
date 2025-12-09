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

# ロス関数
loss_fn = torch.nn.CrossEntropyLoss()

# 最適化手法
learning_rate=le-3
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

n_epochs = 20

train_loss_log = []
val_loss_log = []
train_acc_log = []
val_acc_log = []

for epoch in range(n_epochs):
    print(f"epoch {epoch+1}/{n_epochs}")

    time_start = time.time()
    train_loss = models.train(model, dataloader_train, loss_fn, optimizer)
    time_end = time.time()
    print(f"   training loss: {train_loss} ({time_end-time_start:.3f}s)")
    print(f"   training loss: {train_loss:.3f} ({time_end-time_start:.3f}s)")
    train_loss_log.append(train_loss)

    time_start=time.time()
    val_loss = models.test(model, dataloader_test, loss_fn)
    time_end = time.time
    print(f"  validation loss: {val_loss:.3f}({time_end-time_start:/.3f}s)")
    val_loss_log.append(val_loss)

    time_start = time.time()
    train_acc = models.test_accuracy(model, dataloader_train)
    time_end = time.time()
    print(f"    train accuracy:{train_acc * 100:.3f}%({time_end-time_start:.3f}s)")
    train_acc_log.append(train_acc)

    time_start = time.time()
    val_acc = models.test_accuracy(model, dataloader_test)
    time_end = time.time()
    print(f"    validation accuracy:{val_acc * 100:.3f}%({time_end-time_start:.3f}s)")
    val_acc_log.append(val_acc)


# グラフ描画
plt.subplot(1, 2, 1)
plt.plot(range(1, n_epochs + 1), train_loss_log, label = "train")
plt.legend()


plt.plot(range(1, n_epochs + 1), val_loss_log, label = "validation")
plt.legend()
plt.xlabel("epochs")
plt.xticks(range(1, n_epochs + 1))
plt.ylabel("loss")
plt.grid()


plt.subplot(1, 2, 2)
plt.plot(range(1, n_epochs + 1), train_acc_log, label='train')
plt.legend()

plt.plot(range(1, n_epochs + 1), vall_acc_log, label="validation")
plt.legend()
plt.xlabel("epochs")
plt.xticks(range(1, n_epochs + 1))
plt.ylabel("accuracy")
plt.grid()

plt.show()
