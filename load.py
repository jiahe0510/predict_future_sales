import pandas as pd
from tqdm import tqdm
import os
import numpy as np

_TRAIN_PATH = './data/sales_train.csv'
_ITEM_PATH = './data/items.csv'
_SHOPS_PATH = './data/shops.csv'
_TITLE = './processed_data/train{}.csv'
_TIME_STEP = 34
_ITEM_CNT_MONTH = 'item_cnt_month'

items_file = pd.read_csv(_ITEM_PATH)
shops_file = pd.read_csv(_SHOPS_PATH)
ITEMS_NUM = items_file.shape[0]
SHOPS_NUM = shops_file.shape[0]
_MONTHS = 34


exist_files = []


def process_data_func(path):

    print('{} shops, {} items'.format(SHOPS_NUM, ITEMS_NUM))
    train_file = pd.read_csv(path)
    train_file.sort_values(by=['shop_id'], inplace=True)
    train_file.drop(labels=['date'], axis=1, inplace=True)
    total_price = pd.DataFrame({'total_price': train_file['item_price'] * train_file['item_cnt_day']})

    out = pd.DataFrame(pd.concat([train_file, total_price], axis=1))
    out.drop(labels=['item_price'], axis=1, inplace=True)
    out = out.groupby(by=['shop_id', 'item_id', 'date_block_num']).sum().reset_index()

    single_price = pd.DataFrame({'item_avg_price': out['total_price'] / out['item_cnt_day']})
    out = pd.DataFrame(pd.concat([out, single_price], axis=1))
    out.drop(labels=['total_price'], axis=1, inplace=True)

    out.sort_values(by=['shop_id', 'item_id', 'date_block_num'], inplace=True)
    out = out.rename(index=str, columns={'item_cnt_day': 'item_cnt_month'})
    out.replace([np.inf, -np.inf], 0, inplace=True)

    max_item_cnt, min_item_cnt = out['item_cnt_month'].max(), out['item_cnt_month'].min()
    max_price, min_price = out['item_avg_price'].max(), out['item_avg_price'].min()
    print(max_price, min_price)
    out['item_cnt_month'] = (out['item_cnt_month'] - min_item_cnt) / (max_item_cnt - min_item_cnt) * 20
    out['item_avg_price'] = (out['item_avg_price'] - min_price) / (max_price - min_price)

    return out


def combine_train_data():

    combined_df = pd.DataFrame(
        {'shop_id': [], 'item_id': [], 'date_block_num': [], 'item_cnt_month': [], 'item_avg_price': []})

    for index in range(SHOPS_NUM):
        path = _TITLE.format(index)
        df = pd.read_csv(path)
        combined_df = pd.concat([combined_df, df])

    combined_df.to_csv('./processed_data/train.csv', index=False)


def process_shop_data():

    print('Pre-processing training data...')

    month_list = [i for i in range(_MONTHS)]

    for shop in range(SHOPS_NUM):

        training_data = pd.DataFrame(
            {'shop_id': [], 'item_id': [], 'date_block_num': [], 'item_cnt_month': [], 'item_avg_price': []})

        if 'train'+str(shop)+'.csv' in exist_files:
            print('File train{}.csv exist...'.format(shop))
            continue

        for item in tqdm(range(ITEMS_NUM)):

            check = train.loc[train['shop_id'] == shop]
            check = check.loc[check['item_id'] == item]
            if check.empty:
                continue
            shops = [shop] * _MONTHS
            items = [item] * _MONTHS
            df = pd.DataFrame({'shop_id': shops, 'item_id': items, 'date_block_num': month_list})
            sales = pd.merge(check, df, how='right', on=['shop_id', 'item_id', 'date_block_num'])
            sales.sort_values(by=['date_block_num'], inplace=True)
            sales.fillna(0, inplace=True)
            training_data = pd.concat([training_data, sales])

        training_data.to_csv('./processed_data/train'+str(shop)+'.csv', index=False)


def transfer_np_file(path):

    X, y = None, None

    file = pd.read_csv(path)
    file.drop(labels=['date_block_num'], axis=1, inplace=True)
    shape, start = file.shape[0], 0

    while start < shape:
        end = start + _TIME_STEP
        batch = file.iloc[start: end]
        labels = np.reshape(batch[_ITEM_CNT_MONTH].values, (-1))
        features = np.reshape(batch.drop(labels=[_ITEM_CNT_MONTH], axis=1).values, (-1))
        if X is None:
            X = features
            y = labels
        else:
            X = np.vstack([X, features])
            y = np.vstack([y, labels])
        start += _TIME_STEP
    return X, y


def transfer_csv2npz():
    for shop in tqdm(range(SHOPS_NUM)):
        if 'train_features{}.npz'.format(shop) not in exist_files:
            X_train, y_train = transfer_np_file(_TITLE.format(shop))
            np.save('./processed_data/train_features{}.npy'.format(shop), X_train)
            np.save('./processed_data/train_labels{}.npy'.format(shop), y_train)
        else:
            print('train{}.npz file already exist...'.format(shop))


def process_test_data():
    path1 = './processed_data/train.csv'
    path2 = './data/test.csv'

    all_data = pd.read_csv(path1)
    all_data.drop(labels=['date_block_num'], axis=1, inplace=True)

    test_file = pd.read_csv(path2)
    test_nums = test_file.shape[0]

    def fill_na(s, i):
        shop_id = [s for _ in range(34)]
        item_id = [i for _ in range(34)]
        item_avg_price = [0 for _ in range(34)]
        out = pd.DataFrame({'shop_id': shop_id, 'item_id': item_id, 'item_avg_price': item_avg_price})
        return out

    test_data = None
    for id in tqdm(range(test_nums)):
        part = test_file.loc[test_file['ID'] == id]
        shop = part['shop_id'].values[0]
        item = part['item_id'].values[0]

        check = all_data.loc[all_data['shop_id'] == shop]
        check = check.loc[check['item_id'] == item]
        if check.empty:
            check = fill_na(shop, item)
        else:
            check.drop(labels=['item_cnt_month'], axis=1, inplace=True)
            mean_val = check['item_avg_price'].mean()
            next_line = pd.DataFrame({'shop_id': [shop], 'item_id': [item], 'item_avg_price': [mean_val]})
            check.drop(check.index[0], inplace=True)
            check = pd.concat([check, next_line], ignore_index=True)
        if test_data is None:
            test_data = np.reshape(check.values, (-1))
        else:
            test_data = np.vstack([test_data, np.reshape(check.values, (-1))])

    np.save('./processed_data/test_data.npy', test_data)


if __name__ == '__main__':

    train = process_data_func(_TRAIN_PATH)

    for dirpath, dirnames, files in os.walk('./processed_data/'):
        exist_files = files[:]

    process_shop_data()

    # combine_train_data()

    transfer_csv2npz()

    process_test_data()