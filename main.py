import os
import torch
import argparse
from torch.backends import cudnn
from models.MIMOUNet import build_net
from train import _train
from eval import _eval


def main(args):
    # CUDNN
    cudnn.benchmark = True

    # Create base results directory if it doesn't exist
    base_result_dir = 'results/'
    if not os.path.exists(base_result_dir):
        os.makedirs(base_result_dir)

    # Create model-specific result directories
    model_result_dir = os.path.join(base_result_dir, args.model_name)
    if not os.path.exists(model_result_dir):
        os.makedirs(model_result_dir)

    # Create directories for saving weights and results
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    def load_model(model_path):
        model = build_net('MIMO-UNet')  # Hoặc 'MIMO-UNetPlus'
        state_dict = torch.load(model_path, weights_only=True)
        model.load_state_dict(state_dict['model'])
        model.eval()
        return model

    # Load your model
    model_path = 'results/MIMO-UNet/weights/Final.pkl'
    model = load_model(model_path)

    # Check if CUDA is available
    if torch.cuda.is_available():
        model.cuda()



    # Train or evaluate based on mode
    if args.mode == 'train':
        _train(model, args)
    elif args.mode == 'test':
        _eval(model, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Directories
    parser.add_argument('--model_name', default='MIMO-UNet', choices=['MIMO-UNet', 'MIMO-UNetPlus'], type=str)
    parser.add_argument('--data_dir', type=str, default='dataset/GOPRO')
    parser.add_argument('--mode', default='train', choices=['train', 'test'], type=str)

    # Train mimo-unet
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--num_epoch', type=int, default=24)
    parser.add_argument('--print_freq', type=int, default=30)
    parser.add_argument('--num_worker', type=int, default=4)
    parser.add_argument('--save_freq', type=int, default=30)
    parser.add_argument('--valid_freq', type=int, default=30)
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--lr_steps', type=list, default=[5, 10, 20])

    # Test
    parser.add_argument('--test_model', type=str, default='weights/Final.pkl')
    parser.add_argument('--save_image', type=bool, default=False, choices=[True, False])

    args = parser.parse_args()
    args.model_save_dir = os.path.join('results', args.model_name, 'weights')
    args.result_dir = os.path.join('results', args.model_name, 'result_image')

    print(args)
    main(args)
