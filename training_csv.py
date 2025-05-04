import torch
import torch.nn as nn
import pandas as pd
import os
from torch.profiler import profile, record_function, ProfilerActivity

_model_store: str = 'resource/mymodel.pt'
_training_data: str = 'data/train.csv'

def torch_basic():
    
    # Like pandas and numpy?
    data_1d_list = [1,2,3,4,5]
    data_1d_list_1 = [6,7,8,9,0]
    t = torch.tensor([data_1d_list, data_1d_list_1])

    t = torch.rand(2, 5)
    t = t.to(torch.int)
    t = torch.ones(2, 5)
    print(t)
    print(t.shape)
    print(t.dtype)
    print(t[0][4].item())
    print(t*3)
    print(t.sum().item())
    print(t.sum(axis=1))

    predict = torch.tensor([1, 0, 1])
    label = torch.tensor([0, 1, 2])

    print(torch.mean((predict > label).to(torch.float)))

    print(torch.argsort(predict))
    print(torch.argmax(label))

    data = torch.rand(1, 784)



if __name__ == "__main__":
    print(torch.cuda.is_available())
    # os.environ["CUDA_VISIBLE_DEVICES"] = ''

# prepare the data
    raw_df = pd.read_csv(_training_data)
    # print(raw_df)
    label = raw_df['label'].values
    # print(label)

    raw_df = raw_df.drop(['label'], axis=1)
    feature = raw_df.values
    # print(feature)

    bound = int(len(feature)*0.8)
    train_feature = feature[:bound]
    train_label = label[:bound]
    test_feature = feature[bound:]
    test_label = label[bound:]

    train_feature = torch.tensor(train_feature).to(torch.float).cuda()
    train_label = torch.tensor(train_label).cuda()
    test_feature = torch.tensor(test_feature).to(torch.float).cuda()
    test_label = torch.tensor(test_label).cuda()

    model = nn.Sequential(
        nn.Linear(784, 444),
        nn.ReLU(),
        nn.Linear(444, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 10),
        nn.Softmax()
        ).cuda()

    lossfunction = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(params = model.parameters(), lr=0.0001)
    optimizer = torch.optim.SGD(params = model.parameters())

    # torch.cuda.synchronize()

    with profile(
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True
    ) as prof:
        for i in range(100):
            optimizer.zero_grad()
            predict = model(train_feature)
            # print(predict)
            result = torch.argmax(predict, axis=1)
            # print(result)
            train_cc = torch.mean((result==train_label).to(torch.float))
            # print(train_cc.item())
            loss = lossfunction(predict, train_label)
            loss.backward()
            optimizer.step()
            # print(loss.item())
            print('step {}: train loss: {} train acc: {}'.format(i, loss.item(), train_cc.item()))

            test_predict = model(test_feature)
            test_result = torch.argmax(test_predict, axis=1)
            test_cc = torch.mean((test_result==test_label).to(torch.float))
            test_loss = lossfunction(test_predict, test_label)
            print('step {}: test loss: {} test acc: {}'.format(i, test_loss.item(), test_cc.item()))

            prof.step()

        # print(next(model.parameters()).is_cuda)


    # torch.save(model.state_dict(), _model_store)

    # params = torch.load(_model_store)
    # model.load_state_dict(params)

    # new_test_data = test_feature[100:111]
    # new_test_label = test_label[100:111]
    # new_test_predict = model(new_test_data)
    # new_test_result = torch.argmax(new_test_predict, axis=1)
    # print(new_test_label)
    # print(new_test_result)
    # # test_cc = torch.mean((new_test_result==new_test_label).to(torch.float))


