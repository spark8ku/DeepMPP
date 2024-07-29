import argparse
import os
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d","--data", help="Database to be processed")
    parser.add_argument("-t","--target", help="Target to be predicted")
    parser.add_argument("-n","--network", help="ID of the network")
    parser.add_argument("-l","--load", help="Load the model from the path")
    parser.add_argument("-f","--fragment", help="Index for fragmentation")
    parser.add_argument("-e","--epoch", help="Number of epoch", type=int)
    parser.add_argument("-c","--cuda", help="Cuda device", type=int)
    parser.add_argument("--partial_train",help="Partial training", type=float)
    parser.add_argument("--transfer", help="Model path for transfer learning")

    args = parser.parse_args()
    if args.target:
        args.target = args.target.split(",")
    if args.epoch:
        args.max_epoch = args.epoch
    if args.cuda:
        args.device = "cuda:"+str(args.cuda)
    if args.load:
        args.LOAD_PATH = args.load
        del args.load
        return args
    if args.partial_train:
        if args.partial_train>1:
            args.partial_train = int(args.partial_train)
    if args.transfer:
        args.TRANSFER_PATH = args.transfer
        del args.transfer

    if args.fragment:
        if ',' in args.fragment:
            args.sculptor_index = [int(i) for i in args.fragment.split(',')]
        else:
            args.sculptor_index = [int(i) for i in args.fragment]


    return args
