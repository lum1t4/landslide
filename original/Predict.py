import argparse
import random
import os
import torch
import torch.nn as nn
from torch.utils import data
import torch.backends.cudnn as cudnn
from utils.tools import *
from dataset.landslide_dataset import LandslideDataSet
from model.Networks import Unet
import h5py
import numpy as np
import warnings
warnings.simplefilter("ignore", UserWarning)

img_size = 128 # --> default
n_channels = 3 # --> rgb con ordine brg (estratti da Sentinel-2)
phase = "phase_II_h5_128"
batch_size = 32 # --> default
take_hardcoded_means_stds = False
test_on_training = False
fine_tuning = False
fine_tuning_str = "_fine_tuning" if fine_tuning else ""
model_path = f'./exp_{phase}/batch2000_F1_2160.pth'
generator = torch.Generator()
generator.manual_seed(42)

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Per tutte le GPU
    torch.backends.cudnn.deterministic = True  # Imposta a True per la riproducibilità su CUDA
    torch.backends.cudnn.benchmark = False  # Disabilita l'ottimizzazione specifica dell'hardware
    cudnn.deterministic = True
    cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

def worker_init_fn(worker_id):
    np.random.seed(42 + worker_id)

name_classes = ['Non-Landslide','Landslide']
epsilon = 1e-14

def importName(modulename, name):
    """ Import a named object from a module in the context of this function.
    """
    try:
        module = __import__(modulename, globals(), locals(  ), [name])
    except ImportError:
        return None
    return vars(module)[name]

def get_arguments():

    parser = argparse.ArgumentParser(description="Baseline method for Land4Seen")
    
    parser.add_argument("--data_dir", type=str, default=f'./{phase}/',
                        help="dataset path.")
    parser.add_argument("--model_module", type =str, default='model.Networks',
                        help='model module to import')
    parser.add_argument("--model_name", type=str, default='unet',
                        help='modle name in given module')
    parser.add_argument("--test_list", type=str, default=f'./dataset/test_{phase}.txt',
                        help="test list file.")
    parser.add_argument("--input_size", type=str, default=f'{img_size},{img_size}',
                        help="width and height of input images.")                     
    parser.add_argument("--num_classes", type=int, default=2,
                        help="number of classes.")               
    parser.add_argument("--num_workers", type=int, default=0,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="gpu id in the training.")
    parser.add_argument("--snapshot_dir", type=str, default=f'./test_map_{phase}/',
                        help="where to save predicted maps.")
    parser.add_argument("--restore_from", type=str, default=model_path,
                        help="trained model.")

    return parser.parse_args()


def main():
    args = get_arguments()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    snapshot_dir = args.snapshot_dir
    if os.path.exists(snapshot_dir)==False:
        os.makedirs(snapshot_dir)

    w, h = map(int, args.input_size.split(','))
    input_size = (w, h)

    cudnn.enabled = True
    cudnn.benchmark = True
    
    # Create network   
    model = Unet(n_classes=args.num_classes, n_channels=n_channels)
   
    saved_state_dict = torch.load(args.restore_from, weights_only=True)
    model.load_state_dict(saved_state_dict)

    model = model.cuda()

    test_list = args.train_list if test_on_training else args.test_list
    test_loader = data.DataLoader(
        LandslideDataSet(args.data_dir, test_list, set='unlabeled', channels=n_channels,
                         take_hardcoded_means_stds=take_hardcoded_means_stds),
        batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True,
        worker_init_fn=worker_init_fn)

    # test_loader = data.DataLoader(
    #                 LandslideDataSet(args.data_dir, args.test_list, set='unlabeled'),
    #                 batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)


    interp = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear')
    

    print('Testing..........')
    model.eval()
   

    for index, batch in enumerate(test_loader):  
        image, _, name = batch
        image = image.float().cuda()
        name = name[0].split('.')[0].split('/')[-1].replace('image','mask')
        print(index+1, '/', len(test_loader), ': Testing ', name)  
        
        with torch.no_grad():
            pred = model(image)

        _,pred = torch.max(interp(nn.functional.softmax(pred,dim=1)).detach(), 1)
        pred = pred.squeeze().data.cpu().numpy().astype('uint8')         
        with h5py.File(snapshot_dir+name+'.h5','w') as hf:
            hf.create_dataset('mask', data=pred)

 
if __name__ == '__main__':
    main()
