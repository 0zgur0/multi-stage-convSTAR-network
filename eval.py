
import sys
sys.path.append("models")

import torch.optim
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix as sklearn_cm


def test(model, model_gt, dataloader):
    model.eval()

    logprobabilities = list()
    targets_list = list()
    #inputs_list = list()
    gt_instance_list = list()
    logprobabilities_refined = list()

    for iteration, data in tqdm(enumerate(dataloader)):
        inputs, targets, _, _, gt_instance = data

        if torch.cuda.is_available():
            inputs = inputs.cuda()

        
        y = targets.numpy()
        y_i = gt_instance.cpu().detach().numpy()
        z3, z1, z2 = model.forward(inputs)
        
        z3_refined = model_gt([z1.detach(), z2.detach(), z3.detach()])

        if type(z3_refined) == tuple:
            z3_refined = z3_refined[0]
            
        z1 = z1.cpu().detach().numpy()
        z2 = z2.cpu().detach().numpy()
        z3 = z3.cpu().detach().numpy()
        z3_refined = z3_refined.cpu().detach().numpy()
        
        targets_list.append(y)
        logprobabilities.append(z3)
        gt_instance_list.append(y_i)
        logprobabilities_refined.append(z3_refined)
        
    return np.vstack(logprobabilities), np.concatenate(targets_list), np.vstack(gt_instance_list), np.vstack(logprobabilities_refined)



def confusion_matrix_to_accuraccies(confusion_matrix):

    confusion_matrix = confusion_matrix.astype(float)
    # sum(0) <- predicted sum(1) ground truth

    total = np.sum(confusion_matrix)
    n_classes, _ = confusion_matrix.shape
    overall_accuracy = np.sum(np.diag(confusion_matrix)) / total

    # calculate Cohen Kappa (https://en.wikipedia.org/wiki/Cohen%27s_kappa)
    N = total
    p0 = np.sum(np.diag(confusion_matrix)) / N
    pc = np.sum(np.sum(confusion_matrix, axis=0) * np.sum(confusion_matrix, axis=1)) / N ** 2
    kappa = (p0 - pc) / (1 - pc)

    recall = np.diag(confusion_matrix) / (np.sum(confusion_matrix, axis=1) + 1e-12)
    precision = np.diag(confusion_matrix) / (np.sum(confusion_matrix, axis=0) + 1e-12)
    f1 = (2 * precision * recall) / ((precision + recall) + 1e-12)

    # Per class accuracy
    cl_acc = np.diag(confusion_matrix) / (confusion_matrix.sum(1) + 1e-12)
    
    return overall_accuracy, kappa, precision, recall, f1, cl_acc

def build_confusion_matrix(targets, predictions):
    
    labels = np.unique(targets)
    labels = labels.tolist()
        
    cm = sklearn_cm(targets, predictions, labels=labels)  
    return cm

def print_report(overall_accuracy, kappa, precision, recall, f1, cl_acc):
    
    report="""
    overall accuracy: \t{:.3f}
    kappa \t\t{:.3f}
    precision \t\t{:.3f}
    recall \t\t{:.3f}
    f1 \t\t\t{:.3f}
    """.format(overall_accuracy, kappa, precision.mean(), recall.mean(), f1.mean())

    print(report)
    #print('Per-class acc:', cl_acc)
    return cl_acc



def evaluate_fieldwise(model, model_gt, dataset, batchsize=1, workers=0, viz=False, fold_num=5):
    model.eval()
    model_gt.eval()
    
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batchsize, num_workers=workers)

    logprobabilites, targets, gt_instance, logprobabilites_refined = test(model, model_gt, dataloader)
    predictions = logprobabilites.argmax(1)
    predictions_refined = logprobabilites_refined.argmax(1)
    
    predictions = predictions.flatten()
    targets = targets.flatten()
    gt_instance = gt_instance.flatten()
            
    #Ignore unknown class class_id=0
    if viz:
        valid_crop_samples = targets != 9999999999
    else:
        valid_crop_samples = targets != 0
        
    
    targets_wo_unknown = targets[valid_crop_samples]
    predictions_wo_unknown = predictions[valid_crop_samples]
    gt_instance_wo_unknown = gt_instance[valid_crop_samples]
    predictions_refined_wo_unknown = predictions_refined[valid_crop_samples]

    confusion_matrix = build_confusion_matrix(targets_wo_unknown, predictions_wo_unknown)
    class_acc = print_report(*confusion_matrix_to_accuraccies(confusion_matrix))
    
    print('eval - refined outputs')
    confusion_matrix = build_confusion_matrix(targets_wo_unknown, predictions_refined_wo_unknown)
    print_report(*confusion_matrix_to_accuraccies(confusion_matrix))
    
    pix_acc = np.sum( predictions_wo_unknown==targets_wo_unknown ) / predictions_wo_unknown.shape[0]
    print('Pix acc = %.4f'%pix_acc)

    pix_acc_refined = np.sum( predictions_refined_wo_unknown==targets_wo_unknown ) / predictions_refined_wo_unknown.shape[0]
    print('Pix acc refined = %.4f'%pix_acc_refined)

    prediction_wo_fieldwise = np.zeros_like(targets_wo_unknown)
    num_field = np.unique(gt_instance_wo_unknown).shape[0]
    target_field = np.ones(num_field)*8888
    prediction_field = np.ones(num_field)*9999
    
    count=0
    for i in np.unique(gt_instance_wo_unknown).tolist():
        field_indexes =  gt_instance_wo_unknown==i 
        
        pred = predictions_wo_unknown[field_indexes]
        pred = np.bincount(pred)
        pred =  np.argmax(pred)
        prediction_wo_fieldwise[field_indexes] = pred
        prediction_field[count] = pred 
    
        target = targets_wo_unknown[field_indexes]
        target = np.bincount(target)
        target =  np.argmax(target)
        target_field[count] = target
        count+=1
    
    
    fieldwise_pix_accuracy = np.sum( prediction_wo_fieldwise==targets_wo_unknown ) / prediction_wo_fieldwise.shape[0]
    fieldwise_accuracy = np.sum( prediction_field==target_field ) / prediction_field.shape[0]
    print('Fieldwise pix acc = %.4f'%fieldwise_pix_accuracy)
    print('Fieldwise acc = %.4f'%fieldwise_accuracy)
 
    confusion_matrix = build_confusion_matrix(targets_wo_unknown, prediction_wo_fieldwise)
    print_report(*confusion_matrix_to_accuraccies(confusion_matrix))    
    
    #Save for the visulization 
    if viz:
        predictions = predictions.reshape(-1,24,24)
        prediction_wo_fieldwise = prediction_wo_fieldwise.reshape(-1,24,24)
        targets = targets.reshape(-1,24,24)
        
        np.savez('./viz/result_' + str(fold_num) , targets=targets, predictions=predictions, predictions2=prediction_wo_fieldwise, cm=confusion_matrix)

    else:
        class_labels = dataset.label_list_glob
        class_names = dataset.label_list_glob_name
        existing_class_labels = np.unique(targets)[1:]
    
        for i in range(1,len(class_acc)):
            cur_ind = class_labels.index(existing_class_labels[i])
            name = class_names[int(cur_ind)]
            print(name,' %.4f'%class_acc[i])
    
    
    return fieldwise_pix_accuracy


