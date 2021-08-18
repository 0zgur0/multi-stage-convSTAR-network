import torch
import numpy as np
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import torch.nn
import argparse
import os
from dataset import Dataset
from models.multi_stage_sequenceencoder import multistageSTARSequentialEncoder, multistageLSTMSequentialEncoder
from models.networkConvRef import model_2DConv
from eval import evaluate_fieldwise


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', "--data", type=str, default='/scratch/tmehmet/train_set_24X24_debug.hdf5', help="path to dataset")
    parser.add_argument('-b', "--batchsize", default=4, type=int, help="batch size")
    parser.add_argument('-w', "--workers", default=8, type=int, help="number of dataset worker threads")
    parser.add_argument('-e', "--epochs", default=30, type=int, help="epochs to train")
    parser.add_argument('-l', "--learning_rate", default=0.001, type=float, help="learning rate")
    parser.add_argument('-s', "--snapshot", default=None,
                        type=str, help="load weights from snapshot")
    parser.add_argument('-c', "--checkpoint_dir", default='/home/pf/pfstaff/projects/ozgur_deep_filed/multi_stage/trained_models',
                        type=str,help="directory to save checkpoints")
    parser.add_argument('-wd', "--weight_decay", default=0.0001, type=float, help="weight_decay")
    parser.add_argument('-hd', "--hidden", default=64, type=int, help="hidden dim")
    parser.add_argument('-nl', "--layer", default=6, type=int, help="num layer")
    parser.add_argument('-lrs', "--lrSC", default=2, type=int, help="lrScheduler")
    parser.add_argument('-nm', "--name", default='msConvSTAR', type=str, help="name")
    parser.add_argument('-l1', "--lambda_1", default=0.1, type=float, help="lambda_1")
    parser.add_argument('-l2', "--lambda_2", default=0.5, type=float, help="lambda_2")
    parser.add_argument('-dr', "--dropout", default=0.5, type=float, help="dropout of CNN")
    parser.add_argument('-stg', "--stage", default=3, type=float, help="num stage")
    parser.add_argument('-cp', "--clip", default=5, type=float, help="grad clip")
    parser.add_argument('-sd', "--seed", default=0, type=int, help="random seed")
    parser.add_argument('-fd', "--fold", default=1, type=int, help="5 fold")
    parser.add_argument('-gt', "--gt_path", default='labels.csv', type=str, help="gt file path")
    parser.add_argument('-cell', "--cell", default='star', type=str, help="Cell type: main building block")
    parser.add_argument('-id', "--input_dim", default=4, type=int, help="Input channel size")
    parser.add_argument('-cm', "--apply_cm", default=False, type=bool, help="apply cloud masking")

    return parser.parse_args()


def main(
        datadir=None,
        batchsize=1,
        workers=12,
        epochs=1,
        lr=1e-3,
        snapshot=None,
        checkpoint_dir=None,
        weight_decay=0.0000,
        name='debug',
        layer=6,
        hidden=64,
        lrS=1,
        lambda_1=1,
        lambda_2=1,
        stage=3,
        clip=1,
        fold_num=None,
        gt_path=None,
        cell=None,
        dropout=None,
        input_dim=None,
        apply_cm=None
):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    traindataset = Dataset(datadir, 0., 'train', False, fold_num, gt_path, num_channel=input_dim, apply_cloud_masking=apply_cm)
    testdataset = Dataset(datadir, 0., 'test', True, fold_num, gt_path, num_channel=input_dim, apply_cloud_masking=apply_cm)

    nclasses = traindataset.n_classes
    nclasses_local_1 = traindataset.n_classes_local_1
    nclasses_local_2 = traindataset.n_classes_local_2

    LOSS_WEIGHT = torch.ones(nclasses)
    LOSS_WEIGHT[0] = 0
    LOSS_WEIGHT_LOCAL_1 = torch.ones(nclasses_local_1)
    LOSS_WEIGHT_LOCAL_1[0] = 0
    LOSS_WEIGHT_LOCAL_2 = torch.ones(nclasses_local_2)
    LOSS_WEIGHT_LOCAL_2[0] = 0

    # Class stage mappping
    s1_2_s3 = traindataset.l1_2_g
    s2_2_s3 = traindataset.l2_2_g

    traindataloader = torch.utils.data.DataLoader(traindataset, batch_size=batchsize, shuffle=True, num_workers=workers)

    # Define the model
    if cell == 'lstm':
        network = multistageLSTMSequentialEncoder(24, 24, nstage=stage, nclasses=nclasses,
                                                  nclasses_l1=nclasses_local_1, nclasses_l2=nclasses_local_2,
                                                  input_dim=input_dim, hidden_dim=hidden, n_layers=layer,
                                                  wo_softmax=True)
    else:
        network = multistageSTARSequentialEncoder(24, 24, nstage=stage, nclasses=nclasses,
                                                  nclasses_l1=nclasses_local_1, nclasses_l2=nclasses_local_2,
                                                  input_dim=input_dim, hidden_dim=hidden, n_layers=layer, cell=cell,
                                                  wo_softmax=True)
    network_gt = model_2DConv(nclasses=nclasses, num_classes_l1=nclasses_local_1, num_classes_l2=nclasses_local_2,
                              s1_2_s3=s1_2_s3, s2_2_s3=s2_2_s3,
                              wo_softmax=True, dropout=dropout)

    model_parameters = filter(lambda p: p.requires_grad, network.parameters())
    model_parameters2 = filter(lambda p: p.requires_grad, network_gt.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters]) + sum([np.prod(p.size()) for p in model_parameters2])
    print('Num params: ', params)

    optimizer = torch.optim.Adam(list(network.parameters()) + list(network_gt.parameters()), lr=lr,
                                 weight_decay=weight_decay)

    loss = torch.nn.CrossEntropyLoss(weight=LOSS_WEIGHT)
    loss_local_1 = torch.nn.CrossEntropyLoss(weight=LOSS_WEIGHT_LOCAL_1)
    loss_local_2 = torch.nn.CrossEntropyLoss(weight=LOSS_WEIGHT_LOCAL_2)

    if lrS == 1:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1, last_epoch=-1)
    elif lrS == 2:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1, last_epoch=-1)
    elif lrS == 3:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5, last_epoch=-1)
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5, last_epoch=-1)


    print('CUDA available: ', torch.cuda.is_available())
    if torch.cuda.is_available():
        network = torch.nn.DataParallel(network).cuda()
        network_gt = torch.nn.DataParallel(network_gt).cuda()
        loss = loss.cuda()
        loss_local_1 = loss_local_1.cuda()
        loss_local_2 = loss_local_2.cuda()

    start_epoch = 0
    best_test_acc = 0

    if snapshot is not None:
        checkpoint = torch.load(snapshot)
        network.load_state_dict(checkpoint['network_state_dict'])
        network_gt.load_state_dict(checkpoint['network_gt_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizerA_state_dict'])

    for epoch in range(start_epoch, epochs):
        print("\nEpoch {}".format(epoch+1))

        train_epoch(traindataloader, network, network_gt, optimizer, loss, loss_local_1, loss_local_2,
                    lambda_1=lambda_1, lambda_2=lambda_2, stage=stage, grad_clip=clip)

        # call LR scheduler
        lr_scheduler.step()

        # evaluate model
        if epoch > 1 and epoch % 1 == 0:
            print("\n Eval on test set")
            test_acc = evaluate_fieldwise(network, network_gt, testdataset, batchsize=batchsize)

            if checkpoint_dir is not None:
                checkpoint_name = os.path.join(checkpoint_dir, name + '_epoch_' + str(epoch) + "_model.pth")
                if test_acc > best_test_acc:
                    print('Model saved! Best val acc:', test_acc)
                    best_test_acc = test_acc
                    torch.save({'network_state_dict': network.state_dict(),
                                'network_gt_state_dict': network_gt.state_dict(),
                                'optimizerA_state_dict': optimizer.state_dict()}, checkpoint_name)


def train_epoch(dataloader, network, network_gt, optimizer, loss, loss_local_1, loss_local_2, lambda_1,
                lambda_2, stage, grad_clip):

    network.train()
    network_gt.train()

    mean_loss_glob = 0.
    mean_loss_local_1 = 0.
    mean_loss_local_2 = 0.
    mean_loss_gt = 0.

    for iteration, data in enumerate(dataloader):
        optimizer.zero_grad()

        input, target_glob, target_local_1, target_local_2 = data

        if torch.cuda.is_available():
            input = input.cuda()
            target_glob = target_glob.cuda()
            target_local_1 = target_local_1.cuda()
            target_local_2 = target_local_2.cuda()

        output_glob, output_local_1, output_local_2 = network.forward(input)

        l_glob = loss(output_glob, target_glob)
        l_local_1 = loss_local_1(output_local_1, target_local_1)
        l_local_2 = loss_local_2(output_local_2, target_local_2)

        if stage == 3 or stage == 1:
            total_loss = l_glob + lambda_1 * l_local_1 + lambda_2 * l_local_2
        elif stage == 2:
            total_loss = l_glob + lambda_2 * l_local_2
        else:
            total_loss = l_glob

        mean_loss_glob += l_glob.data.cpu().numpy()
        mean_loss_local_1 += l_local_1.data.cpu().numpy()
        mean_loss_local_2 += l_local_2.data.cpu().numpy()

        # Refinement -------------------------------------------------
        output_glob_R = network_gt([output_local_1, output_local_2, output_glob])
        l_gt = loss(output_glob_R, target_glob)
        mean_loss_gt += l_gt.data.cpu().numpy()

        total_loss = total_loss + l_gt
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), grad_clip)
        torch.nn.utils.clip_grad_norm_(network_gt.parameters(), grad_clip)
        optimizer.step()

    print('Local Loss 1: %.4f' % (mean_loss_local_1 / iteration))
    print('Local Loss 2: %.4f' % (mean_loss_local_2 / iteration))
    print('Global Loss: %.4f' % (mean_loss_glob / iteration))
    print('Global Loss - Refined: %.4f' % (mean_loss_gt / iteration))



if __name__ == "__main__":
    args = parse_args()
    print(args)

    model_name = str(args.name) + '_' + str(args.cell) + '_' + str(args.input_dim) + '_' + str(args.batchsize) + '_' + str(
        args.learning_rate) + '_' + str(args.layer) + '_' + str(args.hidden) + '_' + str(args.lrSC) + '_' + str(
        args.lambda_1) + '_' + str(args.lambda_2) + '_' + str(args.weight_decay) + '_' + str(args.fold) + '_' + str(
        args.gt_path) + '_' + str(args.seed)
    print(model_name)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    main(
        datadir=args.data,
        batchsize=args.batchsize,
        workers=args.workers,
        epochs=args.epochs,
        lr=args.learning_rate,
        snapshot=args.snapshot,
        checkpoint_dir=args.checkpoint_dir,
        weight_decay=args.weight_decay,
        name=model_name,
        layer=args.layer,
        hidden=args.hidden,
        lrS=args.lrSC,
        lambda_1=args.lambda_1,
        lambda_2=args.lambda_2,
        stage=args.stage,
        clip=args.clip,
        fold_num=args.fold,
        gt_path=args.gt_path,
        cell=args.cell,
        dropout=args.dropout,
        input_dim=args.input_dim,
        apply_cm = args.apply_cm
    )