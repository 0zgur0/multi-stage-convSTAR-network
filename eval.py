import sys
sys.path.append("src")
sys.path.append("src/models")
import torch.optim
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix as sklearn_cm

def test(model, model_gt, dataloader, level=3):
    model.eval()

    logprobabilities = list()
    targets_list = list()
    gt_instance_list = list()
    logprobabilities_refined = list()

    for iteration, data in tqdm(enumerate(dataloader)):
        if level==1:
            inputs, _, targets, _, gt_instance = data
        elif level ==2:
            inputs, _, _, targets, gt_instance = data
        else:
            inputs, targets, _, _, gt_instance = data

        del data

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
        gt_instance_list.append(y_i)

        if level==1:
            logprobabilities.append(z1)
        elif level ==2:
            logprobabilities.append(z2)
        else:
            logprobabilities.append(z3)

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
    #nclasses = len(labels)
        
    cm = sklearn_cm(targets, predictions, labels=labels)
#    precision = precision_score(targets, predictions, labels=labels, average='macro')
#    recall = recall_score(targets, predictions, labels=labels, average='macro')
#    f1 = f1_score(targets, predictions, labels=labels, average='macro')
#    kappa = cohen_kappa_score(targets, predictions, labels=labels)
    #print('precision, recall, f1, kappa: ', precision, recall, f1, kappa)
    
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


def evaluate_fieldwise(model, model_gt, dataset, batchsize=1, workers=8, viz=False, fold_num=5, level=3,
                        ignore_undefined_classes=False):
    model.eval()
    model_gt.eval()

    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batchsize, num_workers=workers)

    logprobabilites, targets, gt_instance, logprobabilites_refined = test(model, model_gt, dataloader, level)
    predictions = logprobabilites.argmax(1)
    predictions_refined = logprobabilites_refined.argmax(1)

    predictions = predictions.flatten()
    targets = targets.flatten()
    gt_instance = gt_instance.flatten()
    predictions_refined = predictions_refined.flatten()

    # Ignore unknown class class_id=0
    if viz:
        valid_crop_samples = targets != 9999999999
    elif level == 2 and ignore_undefined_classes:
        valid_crop_samples = (targets != 0) * (targets != 7) * (targets != 9) * (targets != 12)
    elif level == 2:

        targets[(targets == 7)] = 12
        targets[(targets == 9)] = 12
        predictions[(predictions == 7)] = 12
        predictions[(predictions == 9)] = 12
        valid_crop_samples = (targets != 0) * (targets != 7) * (targets != 9)
    else:
        valid_crop_samples = targets != 0

    targets_wo_unknown = targets[valid_crop_samples]
    predictions_wo_unknown = predictions[valid_crop_samples]
    gt_instance_wo_unknown = gt_instance[valid_crop_samples]
    predictions_refined_wo_unknown = predictions_refined[valid_crop_samples]

    labels = np.unique(targets_wo_unknown)
    print('Num class: ', str(labels.shape[0]))

    if level == 3:
        confusion_matrix = build_confusion_matrix(targets_wo_unknown, predictions_refined_wo_unknown)
    else:
        confusion_matrix = build_confusion_matrix(targets_wo_unknown, predictions_wo_unknown)
    print_report(*confusion_matrix_to_accuraccies(confusion_matrix))


    prediction_wo_fieldwise = np.zeros_like(targets_wo_unknown)
    prediction_wo_fieldwise_refined = np.zeros_like(targets_wo_unknown)
    num_field = np.unique(gt_instance_wo_unknown).shape[0]
    target_field = np.ones(num_field) * 8888
    prediction_field = np.ones(num_field) * 9999

    count = 0
    for i in np.unique(gt_instance_wo_unknown).tolist():
        field_indexes = gt_instance_wo_unknown == i

        pred = predictions_wo_unknown[field_indexes]
        pred = np.bincount(pred)
        pred = np.argmax(pred)
        prediction_wo_fieldwise[field_indexes] = pred
        prediction_field[count] = pred

        pred = predictions_refined_wo_unknown[field_indexes]
        pred = np.bincount(pred)
        pred = np.argmax(pred)
        prediction_wo_fieldwise_refined[field_indexes] = pred

        target = targets_wo_unknown[field_indexes]
        target = np.bincount(target)
        target = np.argmax(target)
        target_field[count] = target
        count += 1

    if level == 3:
        confusion_matrix = build_confusion_matrix(targets_wo_unknown, prediction_wo_fieldwise_refined)
    else:
        confusion_matrix = build_confusion_matrix(targets_wo_unknown, prediction_wo_fieldwise)

    print_report(*confusion_matrix_to_accuraccies(confusion_matrix))
    pix_accuracy = np.sum( prediction_wo_fieldwise_refined==targets_wo_unknown ) / prediction_wo_fieldwise_refined.shape[0]

    # Save for the visulization
    if viz:
        prediction_wo_fieldwise = prediction_wo_fieldwise.reshape(-1, 24, 24)
        targets = targets.reshape(-1, 24, 24)

        if level == 3:
            np.savez('/home/pf/pfstaff/projects/ozgur_MSconvRNN/result/msSTAR_ch_analysis4_level_' + str(
                level) + '_fold_' + str(fold_num), targets=targets,
                     predictions_refined=prediction_wo_fieldwise_refined, cm=confusion_matrix,
                     predictions=predictions_refined_wo_unknown)
        else:
            np.savez('/home/pf/pfstaff/projects/ozgur_MSconvRNN/result/msSTAR_ch_analysis4_level_' + str(
                level) + '_fold_' + str(fold_num), targets=targets, predictions=prediction_wo_fieldwise,
                     cm=confusion_matrix)

    return pix_accuracy