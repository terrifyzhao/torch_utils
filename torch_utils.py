import torch
import time
from torchtext import data, datasets
import jieba


class TrainHandler:

    def __init__(self,
                 train_loader,
                 valid_loader,
                 model,
                 criterion,
                 optimizer,
                 model_path,
                 batch_size=32,
                 epochs=5,
                 scheduler=None,
                 gpu_num=0):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.model_path = model_path
        self.batch_size = batch_size
        self.epochs = epochs
        self.scheduler = scheduler
        if torch.cuda.is_available():
            self.device = torch.device(f'cuda:{gpu_num}')
            print('Training device is gpu:{gpu_num}')
        else:
            self.device = torch.device('cpu')
            print('Training device is cpu')
        self.model = model.to(self.device)

    def _train_func(self):
        train_loss = 0
        train_correct = 0
        for i, (x, y) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            x, y = x.to(self.device), y.to(self.device)
            output = self.model(x)
            loss = self.criterion(output, y)
            train_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            train_correct += (output.argmax(1) == y).sum().item()

        if self.scheduler is not None:
            self.scheduler.step()

        return train_loss / len(self.train_loader), train_correct / len(self.train_loader.dataset)

    def _test_func(self):
        valid_loss = 0
        valid_correct = 0
        for x, y in self.valid_loader:
            x, y = x.to(self.device), y.to(self.device)
            with torch.no_grad():
                output = self.model(x)
                loss = self.criterion(output, y)
                valid_loss += loss.item()
                valid_correct += (output.argmax(1) == y).sum().item()

        return valid_loss / len(self.valid_loader), valid_correct / len(self.valid_loader.dateset)

    def train(self):
        min_valid_loss = float('inf')

        for epoch in range(self.epochs):
            start_time = time.time()
            train_loss, train_acc = self._train_func()
            valid_loss, valid_acc = self._test_func()

            if min_valid_loss > valid_loss:
                min_valid_loss = valid_loss
                torch.save(self.model.state_dict(), self.model_path)
                print(f'\tSave model done valid loss: {valid_loss:.4f}')

            secs = int(time.time() - start_time)
            mins = secs / 60
            secs = secs % 60

            print('Epoch: %d' % (epoch + 1), " | time in %d minutes, %d seconds" % (mins, secs))
            print(f'\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc * 100:.1f}%(train)')
            print(f'\tLoss: {valid_loss:.4f}(valid)\t|\tAcc: {valid_acc * 100:.1f}%(valid)')


class TextProcess:
    def __init__(self):
        def tokenizer(text):
            return list(jieba.cut(text))

        TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True, fix_length=20)
        LABEL = data.Field(sequential=False, use_vocab=False)
        all_dataset = data.TabularDataset.splits(path='',
                                                 train='LCQMC.csv',
                                                 format='csv',
                                                 fields=[('sentence1', TEXT), ('sentence2', TEXT), ('label', LABEL)])[0]
        TEXT.build_vocab(all_dataset)
        train, valid = all_dataset.split(0.1)
        (train_iter, valid_iter) = data.BucketIterator.splits(datasets=(train, valid),
                                                              batch_sizes=(64, 128),
                                                              sort_key=lambda x: len(x.sentence1))


if __name__ == '__main__':
    TextProcess()
