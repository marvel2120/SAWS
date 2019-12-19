# read loss file and print
import matplotlib.pyplot as plt
from config import Config

config = Config()
# read from loss file
train, test = [], []
with open(config.train_loss_path) as f:
    for line in f.readlines():
        line = line.strip()
        train.append(float(line[:-1]))
f.close()
with open(config.test_loss_path) as f:
    for line in f.readlines():
        line = line.strip()
        test.append(float(line[:-1]))
f.close()
index = [i for i in range(len(train))]

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss changes with epoch")
plt.plot(index, train, color='green', label='train loss')
plt.plot(index, test, color='red', label='test loss')
plt.legend()
plt.show()
