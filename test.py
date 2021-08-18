import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import torch
import numpy as np
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import torch.nn
import argparse
from dataset import Dataset
from models.multi_stage_sequenceencoder import multistageSTARSequentialEncoder
from models.networkConvRef import model_2DConv
from eval import evaluate_fieldwise


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', "--data", type=str, default='/scratch/tmehmet/train_set_24X24_debug.hdf5', help="path to dataset")
    parser.add_argument('-fd', "--fold", default=1, type=int, help="5 fold")
    parser.add_argument('-b', "--batchsize", default=1, type=int, help="batch size")
    parser.add_argument('-s', "--snapshot",type=str, help="load weights from snapshot",
                        default='/home/pf/pfstaff/projects/ozgur_MSconvRNN/trained_models_rep/fold1.pth',)
    parser.add_argument('-hd', "--hidden", default=64, type=int, help="hidden dim")
    parser.add_argument('-nl', "--layer", default=6, type=int, help="num layer")
    parser.add_argument('-stg', "--stage", default=3, type=float, help="num stage")
    parser.add_argument('-id', "--input_dim", default=4, type=int, help="Input channel size")
    parser.add_argument('-sd', "--seed", default=0, type=int, help="random seed")
    parser.add_argument('-gt', "--gt_path", default='labels.csv', type=str, help="gt file path")
    return parser.parse_args()


def main(
        datadir=None,
        fold_num=None,
        batchsize=1,
        snapshot=None,
        layer=6,
        hidden=64,
        stage=3,
        gt_path=None,
        input_dim=None
):

    testdataset = Dataset(datadir, 0., 'test', eval_mode=True, fold=fold_num, gt_path=gt_path, return_cloud_cover=False,
                          apply_cloud_masking=False, cloud_threshold=0.1, num_channel=input_dim)

    nclasses = testdataset.n_classes
    nclasses_local_1 = testdataset.n_classes_local_1
    nclasses_local_2 = testdataset.n_classes_local_2
    print('Num classes:', nclasses)

    # Class stage mappping
    s1_2_s3 = testdataset.l1_2_g
    s2_2_s3 = testdataset.l2_2_g

    # Define the model
    network = multistageSTARSequentialEncoder(24, 24, nstage=stage, nclasses=nclasses,
                                              nclasses_l1=nclasses_local_1, nclasses_l2=nclasses_local_2,
                                              input_dim=input_dim, hidden_dim=hidden, n_layers=layer, cell='star',
                                              wo_softmax=True)
    network_gt = model_2DConv(nclasses=nclasses, num_classes_l1=nclasses_local_1, num_classes_l2=nclasses_local_2,
                          s1_2_s3=s1_2_s3, s2_2_s3=s2_2_s3,
                          wo_softmax=True, dropout=1)
    network.eval()
    network_gt.eval()

    model_parameters = filter(lambda p: p.requires_grad, network.parameters())
    model_parameters2 = filter(lambda p: p.requires_grad, network_gt.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters]) + sum([np.prod(p.size()) for p in model_parameters2])
    print('Num params: ', params)

    if torch.cuda.is_available():
        network = torch.nn.DataParallel(network).cuda()
        network_gt = torch.nn.DataParallel(network_gt).cuda()

    if snapshot is not None:
        checkpoint = torch.load(snapshot)
        network.load_state_dict(checkpoint['network_state_dict'])
        network_gt.load_state_dict(checkpoint['network_gt_state_dict'])

    evaluate_fieldwise(network, network_gt, testdataset, batchsize=batchsize, level=1, fold_num=fold_num)
    evaluate_fieldwise(network, network_gt, testdataset, batchsize=batchsize, level=2, fold_num=fold_num)
    evaluate_fieldwise(network, network_gt, testdataset, batchsize=batchsize, level=3, fold_num=fold_num)

if __name__ == "__main__":
    args = parse_args()
    print(args)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    main(
        datadir=args.data,
        fold_num=args.fold,
        batchsize=args.batchsize,
        snapshot=args.snapshot,
        layer=args.layer,
        hidden=args.hidden,
        stage=args.stage,
        gt_path=args.gt_path,
        input_dim=args.input_dim
    )