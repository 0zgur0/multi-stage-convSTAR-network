import numpy as np
import datetime
import pandas as pd

#try:
from visdom import Visdom
#except:
#    print("could not find visdom package. try 'pip install visdom'. continue without...")
#    Visdom=None
#    pass

class Printer():

    def __init__(self, batchsize = None, N = None):
        self.batchsize = batchsize
        self.N = N

        self.last=datetime.datetime.now()

    def print(self, stats, iteration):
        print_lst = list()

        if self.N is None:
            print_lst.append('iteration: {}'.format(iteration))
        else:
            print_lst.append('iteration: {}/{}'.format(iteration, self.N))

        dt = (datetime.datetime.now() - self.last).total_seconds()

        print_lst.append('logs/sec: {:.2f}'.format(dt / 1))

        if self.batchsize is not None:
            print_lst.append('samples/sec: {:.2f}'.format(dt / self.batchsize))

        for k, v in zip(stats.keys(), stats.values()):
            print_lst.append('{}: {:.2f}'.format(k, v))

        print('\r' + ', '.join(print_lst), end="")

        self.last = datetime.datetime.now()


class Logger():

    def __init__(self, columns, modes, csv=None, epoch=0, idx=0):

        self.columns=columns
        self.mode=modes[0]
        self.epoch=epoch
        self.idx = idx
        self.data = pd.DataFrame(columns=["epoch","iteration","mode"]+self.columns)
        self.csv = csv

    def resume(self, data):
        self.data = data
        self.idx = data.index[-1]
        self.epoch = data["epoch"].max()

    def update_epoch(self, epoch=None):
        if epoch is None:
            self.epoch+=1
        else:
            self.epoch=epoch

    def set_mode(self,mode):
        self.mode = mode

    def log(self, stats, iteration):

        stats["epoch"] = self.epoch
        stats["iteration"] = iteration
        stats["mode"] = self.mode

        row = pd.DataFrame(stats, index=[self.idx])

        self.data = self.data.append(row, sort=False)
        self.idx +=1

    def get_data(self):
        return self.data

    def save_csv(self, path=None):
        if path is not None:
            self.data.to_csv(path)
        elif self.csv is not None:
            self.data.to_csv(self.csv)
        else:
            raise ValueError("please provide either path argument or initialize Logger() with csv argument")

class VisdomLogger():
    def __init__(self,**kwargs):
        if Visdom is None:
            self.viz = None # do nothing
            return

        self.viz = Visdom(**kwargs)
        self.windows = dict()

        r = np.random.RandomState(1)
        self.colors = r.randint(0,255, size=(255,3))
        self.colors[0] = np.array([1., 1., 1.])
        self.colors[1] = np.array([0. , 0.18431373, 0.65490196]) # ikb blue

    def update(self, data):
        self.plot_epochs(data)
        self.plot_steps(data)

    def plot_steps(self, data, maxplotpoints=50):

        if not self.viz.check_connection():
            return # do nothing

        name="loss"
        win = "c_"+name
        if "c_"+name in self.windows.keys():
            #win = self.windows["c_"+name]
            update = 'new'
        else:
            #win = None  # first log -> new window
            update = None

        for mode in data["mode"].unique():
            d = data.loc[data["mode"] == mode]

            maxiter = d["iteration"].max()

            if len(d) > maxplotpoints:
                d = d.tail(maxplotpoints)
                #d = d.sample(maxplotpoints).sort_index()

            if maxiter > 0:
                frac_epochs = d["epoch"] + d["iteration"] / maxiter
            else:
                frac_epochs = np.array([0])
            #total_iter = d["epoch"]*maxiter + d["iteration"]
            values = d[name]

            opts = dict(
                title=name,
                showlegend=True,
                xlabel='epochs',
                ylabel=name)

            win = self.viz.line(
                X=frac_epochs,
                Y=values,
                name=mode,
                win=win,
                opts=opts,
                update=update
            )
            update = 'insert'

        self.windows["c_"+name] = win

    def plot_epochs(self, data):
        """
        Plots mean of epochs

        :param data:
        :return:
        """
        if self.viz is None:
            return # do nothing

        if not self.viz.check_connection():
            return # do nothing

        data_mean_per_epoch = data.groupby(["mode", "epoch"]).mean()
        cols = data_mean_per_epoch.columns
        modes = data_mean_per_epoch.index.levels[0]

        for name in cols:

             if name in self.windows.keys():
                 win = self.windows[name]
                 update = 'new'
             else:
                 win = name # first log -> new window
                 update = None

             opts = dict(
                 title=name,
                 showlegend=True,
                 xlabel='epochs',
                 ylabel=name)

             for mode in modes:

                 epochs = data_mean_per_epoch[name].loc[mode].index
                 values = data_mean_per_epoch[name].loc[mode]

                 win = self.viz.line(
                     X=epochs,
                     Y=values,
                     name=mode,
                     win=win,
                     opts=opts,
                     update=update
                 )
                 update='insert'

             self.windows[name] = win


    def plot_images(self, target, output):
        if self.viz is None:
            return

        if not self.viz.check_connection():
            return # do nothing


        # log softmax -> softmax
        output = np.exp(output)

        prediction = np.argmax(output, axis=1)

        target = np.swapaxes(self.colors[target], -1, 1)
        prediction = np.swapaxes(self.colors[prediction], -1, 1)

        self.viz.images(target, win="target", opts=dict(title='Target'))
        self.viz.images(prediction, win="predictions", opts=dict(title='Predictions'))

        b, c, h, w = output.shape
        for cl in range(c):
            arr = np.expand_dims(output[:,cl],1)*255
            self.viz.images(arr, win="class"+str(cl), opts=dict(title="class"+str(cl)))

