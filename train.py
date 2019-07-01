import pandas as pd
import numpy as np
import torch.optim as optim
from torch.utils.data import TensorDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import LSTM
from tqdm import tqdm


_TIME_STEP = 34
SHOP_ID = 'shop_id'
ITEM_ID = 'item_id'
TOTAL_PRICE = 'total_price'
ITEM_CNT_MONTH = 'item_cnt_month'
_TITLE_TRAIN_FEATURES = './processed_data/train_features{}.npy'
_TITLE_TRAIN_LABELS = './processed_data/train_labels{}.npy'
_SHOP_NUM = 60
_TEST_LEN = 214200
exist_files = []
EPOCHS = 1

if __name__ == '__main__':

    print('load processed data...')

    X_train, y_train = None, None

    for shop in tqdm(range(_SHOP_NUM)):
        if X_train is None:
            X_train, y_train = np.load(_TITLE_TRAIN_FEATURES.format(shop)), np.load(_TITLE_TRAIN_LABELS.format(shop))
        else:
            X, y = np.load(_TITLE_TRAIN_FEATURES.format(shop)), np.load(_TITLE_TRAIN_LABELS.format(shop))
            X_train, y_train = np.vstack([X_train, X]), np.vstack([y_train, y])

    X_train, y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)

    print(X_train.shape, y_train.shape)

    train_data = TensorDataset(X_train, y_train)
    trainloader = DataLoader(train_data, batch_size=64, shuffle=True)

    net = LSTM()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.MSELoss()
    running_loss = 0.0

    print('Start training...')
    for epoch in range(EPOCHS):
        print('{} epoch begins...'.format(epoch + 1))
        for i, data in enumerate(trainloader):
            net.zero_grad()
            X, y = data
            X = X.view(-1, 34, 3)
            outputs = net(X)
            labels = y[:, -2:-1]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if i % 1000 == 999:
                print("loss is {}...".format(running_loss / 64))
                running_loss = 0.0

    print('Start testing...')

    predict = []
    X_test = torch.tensor(np.load('./processed_data/test_data.npy'), dtype=torch.float32)
    testloader = DataLoader(X_test, batch_size=256, shuffle=False)
    for i, data in enumerate(testloader):
        X = data.view(-1, 34, 3)
        outputs = net(X)
        outputs = outputs.detach().view(-1).numpy()
        predict += outputs.tolist()
        if i % 80 == 79:
            print('{}% test finished...'.format(i * 256 * 100 // _TEST_LEN))
    ID = [i for i in range(_TEST_LEN)]
    df = pd.DataFrame({'ID': ID, 'item_cnt_month': predict})
    print(df.head(20))
    df.to_csv('./data/submission.csv', index=False)






