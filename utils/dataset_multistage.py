import torch.utils.data
import os, glob
import torch
import numpy as np
#import torch.nn.functional as F
import h5py
from matplotlib import pyplot as plt 
import csv

class Dataset(torch.utils.data.Dataset):
    def __init__(self, path, t=0.9, mode='all', eval_mode=False, fold=None, gt_path='labelsC.csv'):
        
        self.data = h5py.File(path, "r")
        self.samples = self.data["data"].shape[0]
        self.max_obs = self.data["data"].shape[1]
        self.spatial = self.data["data"].shape[2:-1]
        #self.n_classes = np.max( self.data["gt"] ) + 1
        self.t = t
        self.augment_rate = 0.66
        self.eval_mode = eval_mode
        self.fold = fold
        
        #Get train/test split
        if self.fold != None:
            print('5fold: ', fold, '  Mode: ', mode)
            self.valid_list = self.split_5fold(mode, self.fold) 
        else:
            self.valid_list = self.split(mode)
        
        self.valid_samples = self.valid_list.shape[0]
        
        #self.dates = self.chooose_dates()[0].tolist()
        #self.dates = self.chooose_dates_2().tolist()
        #self.max_obs = len(self.dates)
        self.max_obs = 71

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
            #if tier_1[i] == 'Vegetation' and tier_4[i] in ['Meadow','Potatoes', 'Pasture', 'Maize', 'Sugar_beets', 'Sunflowers', 'Vegetables', 'Vines', 'Wheat', 'WinterBarley', 'WinterRapeseed', 'WinterWheat']:

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

#        self.tier_2_elements_reduced = list(set(self.label_list_local_1))
#        self.tier_3_elements_reduced = list(set(self.label_list_local_2))
#        self.tier_4_elements_reduced = list(set(self.label_list_glob))
#        self.tier_2_elements_reduced.sort()
#        self.tier_3_elements_reduced.sort()
#        self.tier_4_elements_reduced.sort()
        
        
#        print(self.label_list)
#        print('-'*20)
#        
#        print(self.label_list_local_1)
#        print(self.label_list_local_1_name)        
#        print('-'*20)
#
#        print(self.label_list_local_2)
#        print(self.label_list_local_2_name)                
#        print('-'*20)
#        
#        print(self.label_list_glob)
#        print(self.label_list_glob_name)
            

        #self.n_classes = len(self.label_list)
        self.n_classes = max(self.label_list_glob) + 1
        self.n_classes_local_1 = max(self.label_list_local_1) + 1
        self.n_classes_local_2 = max(self.label_list_local_2) + 1
#        self.n_classes = len(self.tier_4_elements_reduced)
#        self.n_classes_local_1 = len(self.tier_2_elements_reduced)
#        self.n_classes_local_2 = len(self.tier_3_elements_reduced)
        
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
        #for consistency loss---------------------------------------------------------
        
        print('Number of filed instance: ', np.unique(self.data["gt_instance"][...,0]).shape[0])
       
        
    def __len__(self):
        return self.valid_samples

    def __getitem__(self, idx):
                     
        idx = self.valid_list[idx]
        X = self.data["data"][idx]
        target_ = self.data["gt"][idx,...,0]
        if self.eval_mode:
            gt_instance = self.data["gt_instance"][idx,...,0]


        X = np.transpose(X, (0, 3, 1, 2))
        
        X = X[0::2,:4,...]
        #X = X[self.dates,...] 
        
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
        if self.eval_mode:
            gt_instance = torch.from_numpy(gt_instance).float()


        #augmentation
        if self.eval_mode==False and np.random.rand() < self.augment_rate:
            flip_dir  = np.random.randint(3)
            if flip_dir == 0:
                X = X.flip(2)
                target = target.flip(0)
                target_local_1 = target_local_1.flip(0)
                target_local_2 = target_local_2.flip(0)
                if self.eval_mode:                    
                    gt_instance = gt_instance.flip(0)
            elif flip_dir == 1:
                X = X.flip(3)
                target = target.flip(1)
                target_local_1 = target_local_1.flip(1)
                target_local_2 = target_local_2.flip(1)
                if self.eval_mode:                    
                    gt_instance = gt_instance.flip(1)    
            elif flip_dir == 2:
                X = X.flip(2,3)
                target = target.flip(0,1)  
                target_local_1 = target_local_1.flip(0,1)  
                target_local_2 = target_local_2.flip(0,1)  
                if self.eval_mode:                    
                    gt_instance = gt_instance.flip(0,1)

        
        #keep values between 0-1
        X = X * 1e-4
        
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
    traindataset = Dataset(data_path, 0.,'all')
    #traindataset =  Dataset(data_path , 0., 'test', True, 5)    

    traindataset.data_stat()    

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
    
    
    

#    import random
#    from PIL import Image
#
#    
#    gt_colors_r = [0, 0.25, 0.5,1]*10
#    gt_colors_g = [0, 0.25, 0.5, 1]*10
#    gt_colors_b = [0, 0.25, 0.5,1]*10
#
#    random.shuffle(gt_colors_g)
#    random.shuffle(gt_colors_b)
#    
#    for j in range(100):
#        data = dataset[j]
#        x = data[0]
#        y = data[1]
#        y_l1 = data[2]
#        y_l2 = data[3]
#        
#        x = x.numpy()
#        y = y.numpy()
#        y_l1 = y_l1.numpy()
#        y_l2 = y_l2.numpy()
#        
#        y_r = np.zeros([24,24,1])
#        y_g = np.zeros([24,24,1])
#        y_b = np.zeros([24,24,1])
#
#        y_r_l1 = np.zeros([24,24,1])
#        y_g_l1 = np.zeros([24,24,1])
#        y_b_l1 = np.zeros([24,24,1])
#
#        y_r_l2 = np.zeros([24,24,1])
#        y_g_l2 = np.zeros([24,24,1])
#        y_b_l2 = np.zeros([24,24,1])
#        
#        
#        for l in range(36):
#            y_r[y==l] = gt_colors_r[l]
#            y_g[y==l] = gt_colors_g[l]
#            y_b[y==l] = gt_colors_b[l]
#
#            y_r_l1[y_l1==l] = gt_colors_r[l]
#            #y_g_l1[y_l1==l] = gt_colors_g[l]
#            y_b_l1[y_l1==l] = gt_colors_b[l]
#            
#            y_g_l1[y_l1==l] = 1
#            
#            y_r_l2[y_l2==l] = gt_colors_r[l]
#            y_g_l2[y_l2==l] = gt_colors_g[l]
#            #y_b_l2[y_l1==l] = gt_colors_b[l]
#
#            y_b_l2[y_l1==l] = 1
#        
#        y_rgb = np.concatenate((y_r,y_g,y_b), axis=2)
#        y_rgb = y_rgb*255
#        y_rgb = y_rgb.astype(np.uint8)
#        
#        y_rgb_l1 = np.concatenate((y_r_l1,y_g_l1,y_b_l1), axis=2)
#        y_rgb_l1 = y_rgb_l1*255
#        y_rgb_l1 = y_rgb_l1.astype(np.uint8)
#
#        y_rgb_l2 = np.concatenate((y_r_l2,y_g_l2,y_b_l2), axis=2)
#        y_rgb_l2 = y_rgb_l2*255
#        y_rgb_l2 = y_rgb_l2.astype(np.uint8)
#
#        dirx = './images_swiss/' + str(j)
#        
#        if not os.path.exists(dirx):
#            os.makedirs(dirx)
#        
#        for i in range(70):
#            im = x[i,:-1,...]
#            im[0] = im[0]*255/np.max(im[0])
#            im[1] = im[1]*255/np.max(im[1])
#            im[2] = im[2]*255/np.max(im[2])
#
#            im = np.transpose(im, (1, 2, 0))
#            im = im.astype(np.uint8)
#            #print(im)
#            img = Image.fromarray(im, 'RGB')
#            img.save(dirx + '/' +  str(i) + '.png')
#
#
#        img = Image.fromarray(y_rgb, 'RGB')
#        img.save(dirx + '/' + 'y.png')  
#
#        img = Image.fromarray(y_rgb_l1, 'RGB')
#        img.save(dirx + '/' + 'yl1.png')  
#        
#        img = Image.fromarray(y_rgb_l2, 'RGB')
#        img.save(dirx + '/' + 'yl2.png')  
    

    
    import seaborn as sns

    #for i in range(len(traindataset)): 
    idx = [3420, 10331, 1425, 4265]
    
    x1 = traindataset[idx[0]][0]
    x1 = x1[:,:,12,12]
    y1 = traindataset[idx[0]][1]
    y1 = y1[12,12]

    x2 = traindataset[idx[1]][0]
    x2 = x2[:,:,12,12]
    y2 = traindataset[idx[1]][1]
    y2 = y2[12,12]
        
    x1 = torch.clamp(x1,0,1)
    x2 = torch.clamp(x2,0,1)
    
    colordict = {'B04': 'c', 'B03': 'm', 'B02': 'y', 'B08': 'lightcoral'}
    plotbands = ["B02", "B03", "B04",  "B08"]
    
    fig,axs = plt.subplots(2,1, figsize=(8,5))
    fig.subplots_adjust(hspace = 3.2)
                
    sns.despine(offset=6, left=True,ax=axs[0])
    sns.despine(offset=6, left=True,ax=axs[1])
    fig.tight_layout(rect=[0.03, 0.03, 1, 0.97])
    for band in plotbands:
        axs[0].plot(x1[:,plotbands.index(band)], color=colordict[band], label=band)
        axs[0].set_title("Sentinel-2 bands - winter wheat / sugar beet")
        axs[0].set_ylim(0,1.1)
        axs[0].set_label(band)
        #axs[0].set_xlabel('timestamp')
        axs[0].set_ylabel('reflectance')
        axs[0].yaxis.grid(True)
        
        axs[1].plot(x2[:,plotbands.index(band)], color=colordict[band])
        #axs[1].set_title("Sentinel-2 bands - sugar beet")
        axs[1].set_ylim(*axs[0].get_ylim())
        axs[1].set_xlabel('timestamp')
        axs[1].set_ylabel('reflectance')
        axs[1].yaxis.grid(True)

    fig.legend()   
    fig.savefig("node_data.png", dpi=300, format="png")





