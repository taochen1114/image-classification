import argparse

# Parse arguments
parser = argparse.ArgumentParser(description='Image classification model training')

parser.add_argument('-d', '--data-dir', default='./data', 
                    type=str, help='path to images , default: ./data ')
parser.add_argument('--train', default='train.csv', type=str, help='path to train csv (default: train.csv)')
parser.add_argument('--val', default='val.csv', type=str, help='path to val csv (default: val.csv)')
parser.add_argument('--test', default='test.csv', type=str, help='path to test csv (default: test.csv)')
parser.add_argument('--test-mode', default=None, type=str, help='set True for Testing / Inference mode')

parser.add_argument('-j', '--workers', default=8, type=int,
                    help='number of data loading workers (default: 8)')

parser.add_argument('--epochs', default=50, type=int,
                    help='number of total epochs to run (default: 50)')
parser.add_argument('--batch', default=32, type=int,
                    help='set batch size (default: 32)')
parser.add_argument('--lr', default=0.001, type=float,
                    help='set learning rate (default: 0.001)')
parser.add_argument('--dropout', default=0, type=float,
                    help='dropout ratio (default: 0)')
parser.add_argument('--schedule', type=int, nargs='+', default=[30, 40],
                    help='decrease learning rate at these epochs (default: [30, 50])')
parser.add_argument('--gamma', type=float, default=0.9, help='learning rate be multiplied by gamma on schedule (default: 0.9)')
parser.add_argument('--loss', default='CrossEntropyLoss', 
                    help='loss function [CrossEntropyLoss, BCELoss, BCEWithLogitsLoss, f1score] (default: CrossEntropyLoss)')

parser.add_argument('-c', '--checkpoint', default='model_ckpt', type=str,
                    help='path to save model checkpoint (default: model_ckpt)')


# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet50', 
                    help='model arch [resnet18, resnet34, resnet50, mobilenet_v2, inception_v3] (default: resnet50)')
parser.add_argument('--num-classes', default=2, type=int, help='number of classes (default: 2)')

parser.add_argument('--pretrain', action='store_true', help='set for transfer learning')

parser.add_argument('--early-stop',default=10, type=int,
                    help='early stop epoch (default: 10)')

parser.add_argument('--aug', default=None, type=str, help='set True for use auto image augmentation')

def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed
