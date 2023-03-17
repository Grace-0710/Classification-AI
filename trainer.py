from copy import deepcopy

import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

# Training 하는 부분으로 miniBatch에 따라 반복
class Trainer():

    def __init__(self, model, optimizer, crit):
        self.model = model
        self.optimizer = optimizer
        self.crit = crit

        super().__init__()

    def _train(self, x, y, config):
        self.model.train() # 모드 중요

        # Shuffle before begin.
        indices = torch.randperm(x.size(0), device=x.device)
        x = torch.index_select(x, dim=0, index=indices).split(config.batch_size, dim=0)
        # index_select ( input , dim , index , * , out = None ) :  input차원을 따라 텐서를 인덱싱하는 새 텐서를 반환
        y = torch.index_select(y, dim=0, index=indices).split(config.batch_size, dim=0)

        total_loss = 0

        for i, (x_i, y_i) in enumerate(zip(x, y)):
            y_hat_i = self.model(x_i) # Yi = f(Xi)
            
            # crit : __init__에서 미리 할당 ( == Cross entropy == (bs, ) )
            loss_i = self.crit(y_hat_i, y_i.squeeze()) # i번째 loss

            # Initialize the gradients of the model.
            self.optimizer.zero_grad()
            loss_i.backward()

            # optimizer : __init__에서 미리 할당
            self.optimizer.step()

            if config.verbose >= 2:
                print("Train Iteration(%d/%d): loss=%.4e" % (i + 1, len(x), float(loss_i)))

            # Don't forget to detach to prevent memory leak.
            # loss_i = Tensor => float안하고 더하면 Tensor들을 더하는 것이기 때문에 메모리 발생
            total_loss += float(loss_i)

        return total_loss / len(x)

    def _validate(self, x, y, config):
        # Turn evaluation mode on.
        self.model.eval()

        # Turn on the no_grad mode to make more efficintly.
        with torch.no_grad():
            # Shuffle before begin.
            indices = torch.randperm(x.size(0), device=x.device)
            x = torch.index_select(x, dim=0, index=indices).split(config.batch_size, dim=0)
            y = torch.index_select(y, dim=0, index=indices).split(config.batch_size, dim=0)

            total_loss = 0

            for i, (x_i, y_i) in enumerate(zip(x, y)):
                y_hat_i = self.model(x_i)
                loss_i = self.crit(y_hat_i, y_i.squeeze())

                if config.verbose >= 2:
                    print("Valid Iteration(%d/%d): loss=%.4e" % (i + 1, len(x), float(loss_i)))

                total_loss += float(loss_i)

            return total_loss / len(x)

# 학습 시작시,train_data / valid_data 가져온다
# train_data : 2차원 tensor =>
# ex. mnist : |valid_data| == |train_data| = [ (bs, 784), ... ] 리스트형태

    def train(self, train_data, valid_data, config):
        lowest_loss = np.inf
        best_model = None

        for epoch_index in range(config.n_epochs):
            train_loss = self._train(train_data[0], train_data[1], config) # 평균
            valid_loss = self._validate(valid_data[0], valid_data[1], config) # 평균

            # You must use deep copy to take a snapshot of current best weights.
            if valid_loss <= lowest_loss:
                lowest_loss = valid_loss
                best_model = deepcopy(self.model.state_dict())

            print("Epoch(%d/%d): train_loss=%.4e  valid_loss=%.4e  lowest_loss=%.4e" % (
                epoch_index + 1,
                config.n_epochs,
                train_loss,
                valid_loss,
                lowest_loss,
            ))

        # Restore to best model.
        self.model.load_state_dict(best_model)
