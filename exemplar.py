class Exemplar:
    def __init__(self, max_size, total_cls):
        self.val = {}
        self.train = {}
        self.cur_cls = 0
        self.max_size = max_size
        self.total_classes = total_cls

    def update(self, cls_num, train, val):
        train_x, train_y = train
        val_x, val_y = val
        assert self.cur_cls == len(list(self.val.keys()))
        assert self.cur_cls == len(list(self.train.keys()))
        cur_keys = list(set(val_y))
        self.cur_cls += cls_num
        total_store_num = self.max_size / self.cur_cls if self.cur_cls != 0 else max_size
        train_store_num = int(total_store_num * 0.9)
        val_store_num = int(total_store_num * 0.1)
        for key, value in self.val.items():
            self.val[key] = value[:val_store_num]
        for key, value in self.train.items():
            self.train[key] = value[:train_store_num]

        for x, y in zip(val_x, val_y):
            if y not in self.val:
                self.val[y] = [x]
            else:
                if len(self.val[y]) < val_store_num:
                    self.val[y].append(x)
        assert self.cur_cls == len(list(self.val.keys()))
        for key, value in self.val.items():
            assert len(self.val[key]) == val_store_num

        for x, y in zip(train_x, train_y):
            if y not in self.train:
                self.train[y] = [x]
            else:
                if len(self.train[y]) < train_store_num:
                    self.train[y].append(x)
        assert self.cur_cls == len(list(self.train.keys()))
        for key, value in self.train.items():
            assert len(self.train[key]) == train_store_num

    def get_exemplar_train(self):
        exemplar_train_x = []
        exemplar_train_y = []
        for key, value in self.train.items():
            for train_x in value:
                exemplar_train_x.append(train_x)
                exemplar_train_y.append(key)
        return exemplar_train_x, exemplar_train_y

    def get_exemplar_val(self):
        exemplar_val_x = []
        exemplar_val_y = []
        for key, value in self.val.items():
            for val_x in value:
                exemplar_val_x.append(val_x)
                exemplar_val_y.append(key)
        return exemplar_val_x, exemplar_val_y

    def get_cur_cls(self):
        return self.cur_cls
