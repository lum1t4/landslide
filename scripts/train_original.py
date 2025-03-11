import argparse
import random
import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
import torch.backends.cudnn as cudnn
from utils.tools import *
from dataset.landslide_dataset import LandslideDataSet

import warnings
warnings.simplefilter("ignore", UserWarning)

img_size = 128 # --> default
n_channels = 3 # --> rgb con ordine brg (estratti da Sentinel-2)
phase = "phase_II_h5_128"
batch_size = 32 # --> default
take_hardcoded_means_stds = False
test_on_training = False
fine_tuning = False
fine_tuning_from_phase = 1
fine_tuning_str = f"_fine_tuning_{fine_tuning_from_phase}" if fine_tuning else ""

pretrained_model_path = ""
if fine_tuning_from_phase == 1:
    pretrained_model_path = f"./phase_I/best_model__batch4000_F1_5227.pth"
if fine_tuning_from_phase == 2:
    pretrained_model_path = f"./exp_phase_II/batch5000_F1_0.pth" # --> non ci sono modelli buoni per ora
if fine_tuning_from_phase == 3:
    pretrained_model_path = f"./exp_phase_III/best_model__batch500_F1_8437.pth"

best_model_phase_I = f"./phase_I/best_model__batch4000_F1_5227.pth" # vanilla
best_model_phase_II = f"./exp_phase_II/batch2000_F1_4367.pth" # fine_tuning_1,
best_model_phase_III = f"./exp_phase_III/best_model__batch500_F1_8437.pth" # vanilla

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
    parser.add_argument("--model_module", type=str, default='model.Networks',
                        help='model module to import')
    parser.add_argument("--model_name", type=str, default='unet',
                        help='modle name in given module')
    parser.add_argument("--train_list", type=str, default=f'./dataset/train_{phase}.txt',
                        help="training list file.")
    parser.add_argument("--test_list", type=str, default=f'./dataset/test_{phase}.txt',
                        help="test list file.")
    parser.add_argument("--input_size", type=str, default=f'{img_size},{img_size}',
                        help="width and height of input images.")                     
    parser.add_argument("--num_classes", type=int, default=2,
                        help="number of classes.")               
    parser.add_argument("--batch_size", type=int, default=batch_size,
                        help="number of images in each batch.")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="learning rate.")
    parser.add_argument("--num_steps", type=int, default=5000,
                        help="number of training steps.")
    parser.add_argument("--num_steps_stop", type=int, default=5000,
                        help="number of training steps for early stopping.")
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                        help="regularisation parameter for L2-loss.")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="gpu id in the training.")
    parser.add_argument("--snapshot_dir", type=str, default=f'./exp_{phase}{fine_tuning_str}/',
                        help="where to save snapshots of the model.")

    return parser.parse_args()


def main():
    set_random_seed(42)  # Sostituisci con il seme che preferisci
    args = get_arguments()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    snapshot_dir = args.snapshot_dir
    if os.path.exists(snapshot_dir)==False:
        os.makedirs(snapshot_dir)

    w, h = map(int, args.input_size.split(','))
    input_size = (w, h)

    cudnn.enabled = True
    # cudnn.benchmark = True  --> disabilitato per garantire esperimenti deterministici/replicabili
    
    # Create network
    model_import = importName(args.model_module, args.model_name)
    model = model_import(n_classes=args.num_classes, n_channels=n_channels)

    # Caricare il modello preaddestrato
    if fine_tuning:
        if os.path.exists(pretrained_model_path):
            print(f"Caricamento del modello preaddestrato da: {pretrained_model_path}")
            checkpoint = torch.load(pretrained_model_path, weights_only=True)
            model.load_state_dict(checkpoint)
        else:
            print(f"Modello preaddestrato non trovato in: {pretrained_model_path}")

    model.train()
    model = model.cuda()

    src_loader = data.DataLoader(
                    LandslideDataSet(args.data_dir, args.train_list, max_iters=args.num_steps_stop*args.batch_size, set='labeled', channels=n_channels, take_hardcoded_means_stds=take_hardcoded_means_stds),
                    batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True,
                    worker_init_fn=worker_init_fn, generator=generator)

    test_list = args.train_list if test_on_training else args.test_list
    test_loader = data.DataLoader(
                    LandslideDataSet(args.data_dir, test_list, set='labeled', channels=n_channels, take_hardcoded_means_stds=take_hardcoded_means_stds),
                    batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True,
                    worker_init_fn=worker_init_fn)

    optimizer = optim.Adam(model.parameters(),
                        lr=args.learning_rate, weight_decay=args.weight_decay)
    
    interp = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear')
    
    hist = np.zeros((args.num_steps_stop,3))
    F1_best = 0.5    
    cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=255)

    for batch_id, src_data in enumerate(src_loader):
        if batch_id==args.num_steps_stop:
            break
        tem_time = time.time()
        model.train()
        optimizer.zero_grad()
        
        images, labels, _, _ = src_data
        images = images.cuda()      
        pred = model(images)   
        
        pred_interp = interp(pred)
              
        # CE Loss
        labels = labels.cuda().long()
        cross_entropy_loss_value = cross_entropy_loss(pred_interp, labels)
        _, predict_labels = torch.max(pred_interp, 1)
        predict_labels = predict_labels.detach().cpu().numpy()
        labels = labels.cpu().numpy()
        batch_oa = np.sum(predict_labels==labels)*1./len(labels.reshape(-1))
            
        hist[batch_id, 0] = cross_entropy_loss_value.item()
        hist[batch_id, 1] = batch_oa
        
        cross_entropy_loss_value.backward()
        optimizer.step()

        hist[batch_id,-1] = time.time() - tem_time

        # aggiunto

        tp = np.sum((predict_labels == 1) & (labels == 1))  # True Positives
        tn = np.sum((predict_labels == 0) & (labels == 0))  # True Negatives
        fp = np.sum((predict_labels == 1) & (labels == 0))  # False Positives
        fn = np.sum((predict_labels == 0) & (labels == 1))  # False Negatives

        precision = tp / (tp + fp) if (tp + fp) != 0 else 0
        recall = tp / (tp + fn) if (tp + fn) != 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

        # fine

        if (batch_id+1) % 10 == 0:
            print(f'Iter {batch_id + 1}/{args.num_steps} '
                  f'Time: {10 * np.mean(hist[batch_id - 9:batch_id + 1, -1])} '
                  f'Batch_OA = {np.mean(hist[batch_id - 9:batch_id + 1, 1]) * 100} '
                  f'cross_entropy_loss = {np.mean(hist[batch_id - 9:batch_id + 1, 0])} '
                  f'Precision = {precision * 100} Recall = {recall * 100} F1 = {f1_score * 100} '
                  f'TP = {tp} TN = {tn} FP = {fp} FN = {fn}')
            # print('Iter %d/%d Time: %.2f Batch_OA = %.1f cross_entropy_loss = %.3f'%(batch_id+1,args.num_steps,10*np.mean(hist[batch_id-9:batch_id+1,-1]),np.mean(hist[batch_id-9:batch_id+1,1])*100,np.mean(hist[batch_id-9:batch_id+1,0])))
           
        # evaluation per 500 iterations
        if (batch_id+1) % 500 == 0:            
            print('Testing..........')
            model.eval()
            TP_all = np.zeros((args.num_classes, 1))
            FP_all = np.zeros((args.num_classes, 1))
            TN_all = np.zeros((args.num_classes, 1))
            FN_all = np.zeros((args.num_classes, 1))
            n_valid_sample_all = 0
            F1 = np.zeros((args.num_classes, 1))
        
            for _, batch in enumerate(test_loader):  
                image, label,_, name = batch
                label = label.squeeze().numpy()
                image = image.float().cuda()
                
                with torch.no_grad():
                    pred = model(image)

                _,pred = torch.max(interp(nn.functional.softmax(pred,dim=1)).detach(), 1)
                pred = pred.squeeze().data.cpu().numpy()                       
                               
                TP,FP,TN,FN,n_valid_sample = eval_image(pred.reshape(-1),label.reshape(-1),args.num_classes)
                TP_all += TP
                FP_all += FP
                TN_all += TN
                FN_all += FN
                n_valid_sample_all += n_valid_sample

            OA = np.sum(TP_all)*1.0 / n_valid_sample_all

            for i in range(args.num_classes):
                P = TP_all[i]*1.0 / (TP_all[i] + FP_all[i] + epsilon)
                R = TP_all[i]*1.0 / (TP_all[i] + FN_all[i] + epsilon)
                F1[i] = 2.0*P*R / (P + R + epsilon)
                if i==1:
                    print("TP: " + str(np.sum(TP_all[i])))
                    print("TN: " + str(np.sum(TN_all[i])))
                    print("FP: " + str(np.sum(FP_all[i])))
                    print("FN: " + str(np.sum(FN_all[i])))
                print("class " + name_classes[i])
                print('===>' + name_classes[i] + ' Precision: %.2f'%(P * 100))
                print('===>' + name_classes[i] + ' Recall: %.2f'%(R * 100))
                print('===>' + name_classes[i] + ' F1: %.2f'%(F1[i] * 100))

            mF1 = np.mean(F1)            
            print('===> mean F1: %.2f OA: %.2f'%(mF1*100,OA*100))
            best_model_str = "best_model_"
            if F1[1]>F1_best:
                F1_best = F1[1]
                # save the models        
                print('Save Model')                     
                model_name = f'{best_model_str}_batch'+repr(batch_id+1)+'_F1_'+repr(int(F1[1]*10000))+'.pth'
                torch.save(model.state_dict(), os.path.join(
                    snapshot_dir, model_name))
            else:
                model_name = f'batch' + repr(batch_id + 1) + '_F1_' + repr(int(F1[1] * 10000)) + '.pth'
                torch.save(model.state_dict(), os.path.join(
                    snapshot_dir, model_name))
if __name__ == '__main__':
    main()
