import torch.utils.data
import os, glob
import torch
import numpy as np
#import torch.nn.functional as F
import h5py
from matplotlib import pyplot as plt
import csv
import random
from PIL import Image

class Dataset(torch.utils.data.Dataset):
    def __init__(self, path, t=0.9, mode='all', eval_mode=False, fold=None, gt_path='labelsC.csv',
                 time_downsample_factor=2, num_channel=4, apply_cloud_masking=False, cloud_threshold=0.1,
                 return_cloud_cover=False, small_train_set_mode=False):
        
        self.data = h5py.File(path, "r", libver='latest', swmr=True)
        self.samples = self.data["data"].shape[0]
        self.max_obs = self.data["data"].shape[1]
        self.spatial = self.data["data"].shape[2:-1]
        #self.n_classes = np.max( self.data["gt"] ) + 1
        self.t = t
        self.augment_rate = 0.66
        self.eval_mode = eval_mode
        self.fold = fold
        self.num_channel = num_channel
        self.apply_cloud_masking = apply_cloud_masking
        self.cloud_threshold = cloud_threshold
        self.return_cloud_cover = return_cloud_cover

        #Get train/test split
        if small_train_set_mode:
            print('Small training-set mode - fold: ', fold, '  Mode: ', mode)
            self.valid_list = self.split_train_test_23(mode, self.fold)
        elif self.fold != None:
            print('5fold: ', fold, '  Mode: ', mode)
            self.valid_list = self.split_5fold(mode, self.fold) 
        else:
            self.valid_list = self.split(mode)
        
        self.valid_samples = self.valid_list.shape[0]

        self.time_downsample_factor = time_downsample_factor
        self.max_obs = int(142/self.time_downsample_factor)

        gt_path_ = './utils/' + gt_path        
        if not os.path.exists(gt_path_):
            gt_path_ = './'  + gt_path        
            
        file=open(gt_path_, "r")
        tier_1 = []
        tier_2 = []
        tier_3 = []
        tier_4 = []
        reader = csv.reader(file)
        for line in reader:
            tier_1.append(line[-5])
            tier_2.append(line[-4])
            tier_3.append(line[-3])
            tier_4.append(line[-2])
    
        tier_2[0] = '0_unknown'
        tier_3[0] = '0_unknown'
        tier_4[0] = '0_unknown'
    
        self.label_list = []
        for i in range(len(tier_2)):
            if tier_1[i] == 'Vegetation' and tier_4[i] != '':
                self.label_list.append(i)
                
            if tier_2[i] == '':
                tier_2[i] = '0_unknown'
            if tier_3[i] == '':
                tier_3[i] = '0_unknown'
            if tier_4[i] == '':
                tier_4[i] = '0_unknown'
            
                            
        tier_2_elements = list(set(tier_2))
        tier_3_elements = list(set(tier_3))
        tier_4_elements = list(set(tier_4))
        tier_2_elements.sort()
        tier_3_elements.sort()
        tier_4_elements.sort()
            
        tier_2_ = []
        tier_3_ = []
        tier_4_ = []
        for i in range(len(tier_2)):
            tier_2_.append(tier_2_elements.index(tier_2[i]))
            tier_3_.append(tier_3_elements.index(tier_3[i]))
            tier_4_.append(tier_4_elements.index(tier_4[i]))        
        

        self.label_list_local_1 = []
        self.label_list_local_2 = []
        self.label_list_glob = []
        self.label_list_local_1_name = []
        self.label_list_local_2_name = []
        self.label_list_glob_name = []
        for gt in self.label_list:
            self.label_list_local_1.append(tier_2_[int(gt)])
            self.label_list_local_2.append(tier_3_[int(gt)])
            self.label_list_glob.append(tier_4_[int(gt)])
            
            self.label_list_local_1_name.append(tier_2[int(gt)])
            self.label_list_local_2_name.append(tier_3[int(gt)])
            self.label_list_glob_name.append(tier_4[int(gt)])

        self.n_classes = max(self.label_list_glob) + 1
        self.n_classes_local_1 = max(self.label_list_local_1) + 1
        self.n_classes_local_2 = max(self.label_list_local_2) + 1

        
        print('Dataset size: ', self.samples)
        print('Valid dataset size: ', self.valid_samples)
        print('Sequence length: ', self.max_obs)
        print('Spatial size: ', self.spatial)
        print('Number of classes: ', self.n_classes)
        print('Number of classes - local-1: ', self.n_classes_local_1)
        print('Number of classes - local-2: ', self.n_classes_local_2)

        #for consistency loss---------------------------------------------------------
        self.l1_2_g = np.zeros(self.n_classes)
        self.l2_2_g = np.zeros(self.n_classes)
        self.l1_2_l2 = np.zeros(self.n_classes_local_2)
        
        for i in range(1,self.n_classes):
            if i in self.label_list_glob:
                self.l1_2_g[i] = self.label_list_local_1[self.label_list_glob.index(i)]
                self.l2_2_g[i] = self.label_list_local_2[self.label_list_glob.index(i)]
        
        for i in range(1,self.n_classes_local_2):
            if i in self.label_list_local_2:
                self.l1_2_l2[i] = self.label_list_local_1[self.label_list_local_2.index(i)]

        
    def __len__(self):
        return self.valid_samples

    def __getitem__(self, idx):
                     
        idx = self.valid_list[idx]
        X = self.data["data"][idx]

        if self.apply_cloud_masking or self.return_cloud_cover:
            CC = self.data["cloud_cover"][idx]

        target_ = self.data["gt"][idx,...,0]
        if self.eval_mode:
            gt_instance = self.data["gt_instance"][idx,...,0]

        X = np.transpose(X, (0, 3, 1, 2))

        # Temporal downsampling
        X = X[0::self.time_downsample_factor,:self.num_channel,...]

        if self.apply_cloud_masking or self.return_cloud_cover:
            CC = CC[0::self.time_downsample_factor,...]

        #Change labels 
        target = np.zeros_like(target_)
        target_local_1 = np.zeros_like(target_)
        target_local_2 = np.zeros_like(target_)
        for i in range(len(self.label_list)):
            #target[target_ == self.label_list[i]] = i
#            target[target_ == self.label_list[i]] = self.tier_4_elements_reduced.index(self.label_list_glob[i])
#            target_local_1[target_ == self.label_list[i]] = self.tier_2_elements_reduced.index(self.label_list_local_1[i])
#            target_local_2[target_ == self.label_list[i]] = self.tier_3_elements_reduced.index(self.label_list_local_2[i])
            target[target_ == self.label_list[i]] = self.label_list_glob[i]
            target_local_1[target_ == self.label_list[i]] = self.label_list_local_1[i]
            target_local_2[target_ == self.label_list[i]] = self.label_list_local_2[i]
            
        
        X = torch.from_numpy(X)
        target = torch.from_numpy(target).float()
        target_local_1 = torch.from_numpy(target_local_1).float()
        target_local_2 = torch.from_numpy(target_local_2).float()

        if self.apply_cloud_masking or self.return_cloud_cover:
            CC = torch.from_numpy(CC).float()

        if self.eval_mode:
            gt_instance = torch.from_numpy(gt_instance).float()


        #keep values between 0-1
        X = X * 1e-4

        # Cloud masking
        if self.apply_cloud_masking:
            CC_mask = CC < self.cloud_threshold
            CC_mask = CC_mask.view(CC_mask.shape[0],1,CC_mask.shape[1],CC_mask.shape[2])
            X = X * CC_mask.float()


        #augmentation
        if self.eval_mode==False and np.random.rand() < self.augment_rate:
            flip_dir  = np.random.randint(3)
            if flip_dir == 0:
                X = X.flip(2)
                target = target.flip(0)
                target_local_1 = target_local_1.flip(0)
                target_local_2 = target_local_2.flip(0)
            elif flip_dir == 1:
                X = X.flip(3)
                target = target.flip(1)
                target_local_1 = target_local_1.flip(1)
                target_local_2 = target_local_2.flip(1)
            elif flip_dir == 2:
                X = X.flip(2,3)
                target = target.flip(0,1)  
                target_local_1 = target_local_1.flip(0,1)  
                target_local_2 = target_local_2.flip(0,1)



        if self.return_cloud_cover:
            if self.eval_mode:
                return X.float(), target.long(), target_local_1.long(), target_local_2.long(), gt_instance.long(), CC.float()
            else:
                return X.float(), target.long(), target_local_1.long(), target_local_2.long(), CC.float()
        else:
            if self.eval_mode:
                return X.float(), target.long(), target_local_1.long(), target_local_2.long(), gt_instance.long()
            else:
                return X.float(), target.long(), target_local_1.long(), target_local_2.long()



    def get_rid_small_fg_tiles(self):
        valid = np.ones(self.samples)
        w,h = self.data["gt"][0,...,0].shape
        for i in range(self.samples):
            if np.sum( self.data["gt"][i,...,0] != 0 )/(w*h) < self.t:
                valid[i] = 0
        
        return np.nonzero(valid)[0]
        
    def split(self, mode):
        valid = np.zeros(self.samples)
        if mode=='test':
            valid[int(self.samples*0.75):] = 1.
        elif mode=='train':
            valid[:int(self.samples*0.75)] = 1.
        else:
            valid[:] = 1.

        w,h = self.data["gt"][0,...,0].shape
        for i in range(self.samples):
            if np.sum( self.data["gt"][i,...,0] != 0 )/(w*h) < self.t:
                valid[i] = 0
        
        return np.nonzero(valid)[0]

    def split_5fold(self, mode, fold):
        
        if fold == 1:
            test_s = int(0)
            test_f = int(self.samples*0.2)
        elif fold == 2:
            test_s = int(self.samples*0.2)
            test_f = int(self.samples*0.4)
        elif fold == 3:
            test_s = int(self.samples*0.4)
            test_f = int(self.samples*0.6)
        elif fold == 4:
            test_s = int(self.samples*0.6)
            test_f = int(self.samples*0.8)
        elif fold == 5:
            test_s = int(self.samples*0.8)
            test_f = int(self.samples)            
                     
        if mode=='test':
            valid = np.zeros(self.samples)
            valid[test_s:test_f] = 1.
        elif mode=='train':
            valid = np.ones(self.samples)
            valid[test_s:test_f] = 0.


        w,h = self.data["gt"][0,...,0].shape
        for i in range(self.samples):
            if np.sum( self.data["gt"][i,...,0] != 0 )/(w*h) < self.t:
                valid[i] = 0
        
        return np.nonzero(valid)[0]

    def split_train_test_23(self, mode, fold):

        if fold == 1:
            train_s = int(0)
            train_f = int(self.samples * 0.4)
        elif fold == 2:
            train_s = int(self.samples * 0.2)
            train_f = int(self.samples * 0.6)
        elif fold == 3:
            train_s = int(self.samples * 0.4)
            train_f = int(self.samples * 0.8)
        elif fold == 4:
            train_s = int(self.samples * 0.6)
            train_f = int(self.samples * 1.0)

        if mode == 'test':
            valid = np.ones(self.samples)
            valid[train_s:train_f] = 0.
        elif mode == 'train':
            valid = np.zeros(self.samples)
            valid[train_s:train_f] = 1.

        w, h = self.data["gt"][0, ..., 0].shape
        for i in range(self.samples):
            if np.sum(self.data["gt"][i, ..., 0] != 0) / (w * h) < self.t:
                valid[i] = 0

        return np.nonzero(valid)[0]

    
    def chooose_dates(self):
        #chosen_dates = np.zeros(self.max_obs)
        #samples = np.random.randint(self.samples, size=1000)
        #samples = np.sort(samples)
        samples = self.data["cloud_cover"][0::10,...]
        samples = np.mean(samples, axis=(0,2,3))
        #print(np.nonzero(samples<0.1))
        return np.nonzero(samples<0.1)

    def chooose_dates_2(self):
        data_dir = '/home/pf/pfstaff/projects/ozgur_deep_filed/data_crop_CH/train_set_24x24/'
        DATA_YEAR = '2019'
        date_list = []
        batch_dirs = os.listdir(data_dir)
        for batch_count, batch in enumerate(batch_dirs):
            for filename in glob.iglob(data_dir + batch + '/**/patches_res_R10m.npz', recursive=True):
                    date = filename.find(DATA_YEAR)
                    date = filename[date:date+8]
                    if date not in date_list:
                        date_list.append(date)
        
        dates_text_file = open("./dates_1.txt", "r")
        specific_dates = dates_text_file.readlines()

        print('Number of dates: ', len(specific_dates))
        specific_date_indexes = np.zeros(len(specific_dates))
        for i in range(len(specific_dates)):
            specific_date_indexes[i] = date_list.index(specific_dates[i][:-1])
            
        return specific_date_indexes.astype(int)

    def data_stat(self):
        class_labels = np.unique(self.label_list_glob)
        class_names = np.unique(self.label_list_glob_name)
        class_fq = np.zeros_like(class_labels)
        
        for i in range(self.__len__()): 
            temp = self.__getitem__(i)[1].flatten()
    
            for j in range(class_labels.shape[0]):
               class_fq[j] += torch.sum(temp==class_labels[j]) 

        for x in class_names:
            print(x)
            
        for x in class_fq:
            print(x)


if __name__=="__main__":

    data_path = "/home/pf/pfstaff/projects/ozgur_deep_filed/data_crop_CH/train_set_24x24_debug.hdf5"
    data_path2 = "/home/pf/pfstaff/projects/ozgur_deep_filed/data_crop_CH/train_set_24x24_debug.hdf5"
    traindataset = Dataset(data_path, 0.,'train',eval_mode=True, num_channel=9)
    traindataset2 =  Dataset(data_path2 , 0.,'train',eval_mode=True)

    #traindataset.data_stat()

    nclasses = traindataset.n_classes
    nclasses_local_1 = traindataset.n_classes_local_1
    nclasses_local_2 = traindataset.n_classes_local_2
    
#    #Compute the class frequencies
#    class_fq  = np.zeros(nclasses)
#    class_fq_lolal_1  = np.zeros(nclasses_local_1)
#    class_fq_lolal_2  = np.zeros(nclasses_local_2)

#    for i in range(len(traindataset)): 
#        temp = traindataset[i][1].flatten()
#        temp1 = traindataset[i][2].flatten()
#        temp2 = traindataset[i][3].flatten()
#    
#        for j in range(nclasses):
#           class_fq[j] = class_fq[j] + torch.sum(temp==j) 
#        for j in range(nclasses_local_1):
#           class_fq_lolal_1[j] = class_fq_lolal_1[j] + torch.sum(temp1==j) 
#        for j in range(nclasses_local_2):
#           class_fq_lolal_2[j] = class_fq_lolal_2[j] + torch.sum(temp2==j) 

    
#    class_fq = class_fq.numpy()
#    class_fq_lolal_1 = class_fq_lolal_1.numpy()
#    class_fq_lolal_1 = class_fq_lolal_1.numpy()

#    print(class_fq)
#    print(class_fq_lolal_1)
#    print(class_fq_lolal_1)
#



#    print(traindataset.label_list_glob)
#    print(np.unique(traindataset.label_list_local_2))
#    print(len(np.unique(traindataset.label_list_local_2)))
#
#    with open('class_names.csv', 'w', newline='') as myfile:
#         wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
#         wr.writerow(class_names)
#
#    with open('class_freq.csv', 'w', newline='') as myfile:
#         wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
#         wr.writerow(class_fq)
#
#    with open('class_freq_1.csv', 'w', newline='') as myfile:
#         wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
#         wr.writerow(class_fq_lolal_1)
#
#    with open('class_freq_2.csv', 'w', newline='') as myfile:
#         wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
#         wr.writerow(class_fq_lolal_2)

#    plt.bar(bins[:-1],class_fq[1:])
#    plt.title("density") 
#    plt.savefig('train_gt_density_TG19.png')
#    plt.close()


  
    
#
#    path = './multi_stage_labels.csv'
#    file=open(path, "r")
#    tier_2 = []
#    tier_3 = []
#    tier_4 = []
#    reader = csv.reader(file)
#    for line in reader:
#        tier_2.append(line[-4])
#        tier_3.append(line[-3])
#        tier_4.append(line[-2])
#
#    tier_2[0] = '0_unknown'
#    tier_3[0] = '0_unknown'
#    tier_4[0] = '0_unknown'
#
#    for i in range(len(tier_2)):
#        if tier_2[i] == '':
#            tier_2[i] = '0_unknown'
#        if tier_3[i] == '':
#            tier_3[i] = '0_unknown'
#        if tier_4[i] == '':
#            tier_4[i] = '0_unknown'
#                        
#    tier_2_elements = list(set(tier_2))
#    tier_3_elements = list(set(tier_3))
#    tier_4_elements = list(set(tier_4))
#    tier_2_elements.sort()
#    tier_3_elements.sort()
#    tier_4_elements.sort()
#
#    tier_2_ = []
#    tier_3_ = []
#    tier_4_ = []
#    for i in range(len(tier_2)):
#        tier_2_.append(tier_2_elements.index(tier_2[i]))
#        tier_3_.append(tier_3_elements.index(tier_3[i]))
#        tier_4_.append(tier_4_elements.index(tier_4[i]))
#
#
#    print(tier_4_)
#    print(len(tier_4_))
#    print(tier_4_elements)
#    print(tier_4)
    
    
#    for i in range(len(traindataset)):
#        counts = np.bincount(traindataset[i][1].flatten())
#        print(np.argmax(counts))
#        #print(traindataset[-1][0])
    
#    #Dates 
#    idxes = traindataset.chooose_dates()[0]
#    data_dir = '/home/pf/pfstaff/projects/ozgur_deep_filed/data_crop_CH/train_set_24x24/'
#    DATA_YEAR = '2019'
#    date_list = []
#    batch_dirs = os.listdir(data_dir)
#    for batch_count, batch in enumerate(batch_dirs):
#        for filename in glob.iglob(data_dir + batch + '/**/patches_res_R10m.npz', recursive=True):
#                date = filename.find(DATA_YEAR)
#                date = filename[date:date+8]
#                if date not in date_list:
#                    date_list.append(date) 
#    
#    print("Num dates:", idxes.shape[0])
#    date_list.sort()
#    for i in range(idxes.shape[0]):
#        print(date_list[idxes[i]])
    

    #-----------------------------------------------------------------------------------------------------------
    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
              26, 27, 28, 29, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 48, 49, 50, 51]

    label_names = ['Unknown', 'Apples', 'Beets', 'Berries', 'Biodiversity area', 'Buckwheat',
                   'Chestnut', 'Chicory', 'Einkorn wheat', 'Fallow', 'Field bean', 'Forest',
                   'Gardens', 'Grain', 'Hedge', 'Hemp', 'Hops', 'Legumes', 'Linen', 'Lupine',
                   'Maize', 'Meadow', 'Mixed crop', 'Multiple', 'Mustard', 'Oat', 'Pasture', 'Pears',
                   'Peas', 'Potatoes', 'Pumpkin', 'Rye', 'Sorghum', 'Soy', 'Spelt', 'Stone fruit',
                   'Sugar beet', 'Summer barley', 'Summer rapeseed', 'Summer wheat', 'Sunflowers',
                   'Tobacco', 'Tree crop', 'Vegetables', 'Vines', 'Wheat', 'Winter barley',
                   'Winter rapeseed', 'Winter wheat']

    # for x in labels:
    #     print(x, ' : ' ,label_names[labels.index(x)])

    colordict = {'Unknown': [255, 255, 255],
                 'Apples': [128, 0, 0],
                 'Beets': [238, 232, 170],
                 'Berries': [255, 107, 70],
                 'Biodiversity area': [0, 191, 255],
                 'Buckwheat': [135, 206, 235],
                 'Chestnut': [0, 0, 128],
                 'Chicory': [138, 43, 226],
                 'Einkorn wheat': [255, 105, 180],
                 'Fallow': [0, 255, 255],
                 'Field bean': [210, 105, 30],
                 'Forest': [65, 105, 225],
                 'Gardens': [255, 140, 0],
                 'Grain': [139, 0, 139],
                 'Hedge': [95, 158, 160],
                 'Hemp': [128, 128, 128],
                 'Hops': [147, 112, 219],
                 'Legumes': [85, 107, 47],
                 'Linen': [176, 196, 222],
                 'Lupine': [127, 255, 212],
                 'Maize': [100, 149, 237],
                 'Meadow': [240, 128, 128],
                 'Mixed crop': [255, 99, 71],
                 'Multiple': [220, 220, 220],
                 'Mustard': [0, 128, 128],
                 'Oat': [0, 206, 209],
                 'Pasture': [106, 90, 205],
                 'Pears': [205, 92, 92],
                 'Peas': [186, 85, 211],
                 'Potatoes': [189, 183, 107],
                 'Pumpkin': [34, 139, 34],
                 'Rye': [184, 134, 11],
                 'Sorghum': [0, 100, 0],
                 'Soy': [199, 21, 133],
                 'Spelt': [25, 25, 112],
                 'Stone fruit': [0, 0, 0],
                 'Sugar beet': [152, 251, 152],
                 'Summer barley': [245, 222, 179],
                 'Summer rapeseed': [32, 178, 170],
                 'Summer wheat': [255, 69, 0],
                 'Sunflowers': [0, 0, 255],
                 'Tobacco': [220, 20, 60],
                 'Tree crop': [255, 255, 102],
                 'Vegetables': [255, 20, 147],
                 'Vines': [0, 20, 200],
                 'Wheat': [255, 215, 0],
                 'Winter barley': [128, 128, 0],
                 'Winter rapeseed': [154, 205, 50],
                 'Winter wheat': [124, 252, 0]}


    # for j in range(0, 10000,100):
    #    data = traindataset[j]
    #    data2 = traindataset2[j]
    #    x = data[0]
    #    y = data[1]
    #
    #    x2 = data2[0]
    #    y2 = data2[1]
    #
    #    x = x.numpy()
    #    y = y.numpy()
    #    x2 = x2.numpy()
    #    y2 = y2.numpy()
    #
    #    #print(np.sum(x==x2)/np.prod(x.shape), np.sum(y==y2)/np.prod(y.shape))
    #
    #
    #    y_r = np.zeros([24,24,1])
    #    y_g = np.zeros([24,24,1])
    #    y_b = np.zeros([24,24,1])
    #
    #
    #    for l in range(len(labels)):
    #        y_r[y==labels[l]] = colordict[label_names[l]][0]
    #        y_g[y==labels[l]] = colordict[label_names[l]][1]
    #        y_b[y==labels[l]] = colordict[label_names[l]][2]
    #
    #    y_rgb = np.concatenate((y_r,y_g,y_b), axis=2)
    #    y_rgb = y_rgb.astype(np.uint8)
    #
    #    dirx = './false_image_swiss/' + str(j)
    #
    #    if not os.path.exists(dirx):
    #        os.makedirs(dirx)
    #
    #    max_val = np.max(x[:,0:3,...])
    #    max_val_nir = np.max(x[:,-1,...])
    #
    #    for i in range(0,70,3):
    #        im = x[i,:,...]
    #        b = im[2:3]*255
    #        g = im[1:2]*255
    #        r = im[3:4]*255
    #        nir = im[0:1]*255
    #
    #        im = np.concatenate([r,g,b],0)
    #
    #        im = np.repeat(im, 5, axis=1)
    #        im = np.repeat(im, 5, axis=2)
    #        im = np.transpose(im, (1, 2, 0))
    #
    #        #nir = x[i,-1,...]/np.max(x[i,-1,...])
    #        #nir = np.repeat(nir, 5, axis=0)
    #        #nir = np.repeat(nir, 5, axis=1)
    #
    #        #cm = plt.get_cmap('magma')
    #        #img = Image.fromarray(cm(nir, bytes=True))
    #
    #        false_image = Image.fromarray(np.uint8(im), mode='RGB')
    #        false_image.save(dirx + '/' +  str(i) + '.png')
    #
    #
    #    y_rgb = np.repeat(y_rgb, 5, axis=0)
    #    y_rgb = np.repeat(y_rgb, 5, axis=1)
    #
    #    img = Image.fromarray(y_rgb, 'RGB')
    #    img.save('./false_image_swiss/' + 'y_'+  str(j) +'.png')


    # Image interpretation - Konrad
    num_samples = 24
    labels_des = [20,21,51,27,38,49,50,45]
    X = np.zeros((num_samples,71,9))
    Y = list()
    count = 0

    for i in range(0,10000):
        data = traindataset[i]
        x = data[0]
        y = data[1]

        x = x.view(x.shape[0],x.shape[1],-1)
        y = y.view(-1)

        x = x.numpy()
        y = y.numpy()

        for j in range(y.shape[0]):
            if y[j] in labels_des:
                X[count] = x[:,:,j]
                Y.append( label_names[labels.index(y[j])] )
                count += 1
                labels_des.remove(y[j])
                break

        if count == num_samples:
            break

        if len(labels_des)==0:
            labels_des = [20, 21, 51, 27, 38, 49, 50, 45]
            continue


    Y = np.array(Y)
    inds = Y.argsort()
    Y_sorted = Y[inds]
    X_sorted = X[inds]
    X_sorted = X_sorted*1000

    #np.savez('II_course_crop_data_samples', x=X_sorted, y=Y_sorted)

    #data = np.load('II_course_crop_data_samples.npz')
    #X = data['x']
    #y = data['y']

    print(X_sorted.shape)
    print(X_sorted[0,0,:])
    print(X_sorted[0,1,:])
    print('xxxxx')


    X_sorted_ = X_sorted.reshape((24, 71*9))
    np.savetxt('II_course_crop_data.csv', X_sorted_, delimiter=',')

    print(X_sorted_[0])

    import csv
    Ylist =[]
    for x in Y_sorted.tolist():
        Ylist.append([x])

    file = open('II_course_crop_label.csv', 'w+', newline='')
    with file:
        write = csv.writer(file)
        write.writerows(Ylist)




    # Read data
    X = np.genfromtxt('II_course_crop_data.csv', delimiter=',')
    X = X.reshape((24, 71, 9))

    # Read labels
    Y = list()
    with open('II_course_crop_label.csv', 'r') as read_obj:
        csv_reader = csv.reader(read_obj)
        for l in csv_reader:
            Y.append(l[0])





    # import seaborn as sns
    #
    # #for i in range(len(traindataset)):
    # idx = [3420, 10331, 1425, 4265]
    #
    # x1 = traindataset[idx[0]][0]
    # x1 = x1[:,:,12,12]
    # y1 = traindataset[idx[0]][1]
    # y1 = y1[12,12]
    #
    # x2 = traindataset[idx[1]][0]
    # x2 = x2[:,:,12,12]
    # y2 = traindataset[idx[1]][1]
    # y2 = y2[12,12]
    #
    # x1 = torch.clamp(x1,0,1)
    # x2 = torch.clamp(x2,0,1)
    #
    # colordict = {'B04': 'c', 'B03': 'm', 'B02': 'y', 'B08': 'lightcoral'}
    # plotbands = ["B02", "B03", "B04",  "B08"]
    #
    # fig,axs = plt.subplots(2,1, figsize=(8,5))
    # fig.subplots_adjust(hspace = 3.2)
    #
    # sns.despine(offset=6, left=True,ax=axs[0])
    # sns.despine(offset=6, left=True,ax=axs[1])
    # fig.tight_layout(rect=[0.03, 0.03, 1, 0.97])
    # for band in plotbands:
    #     axs[0].plot(x1[:,plotbands.index(band)], color=colordict[band], label=band)
    #     axs[0].set_title("Sentinel-2 bands - winter wheat / sugar beet")
    #     axs[0].set_ylim(0,1.1)
    #     axs[0].set_label(band)
    #     #axs[0].set_xlabel('timestamp')
    #     axs[0].set_ylabel('reflectance')
    #     axs[0].yaxis.grid(True)
    #
    #     axs[1].plot(x2[:,plotbands.index(band)], color=colordict[band])
    #     #axs[1].set_title("Sentinel-2 bands - sugar beet")
    #     axs[1].set_ylim(*axs[0].get_ylim())
    #     axs[1].set_xlabel('timestamp')