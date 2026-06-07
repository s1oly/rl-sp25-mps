import torch
from torch import nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from absl import app, flags
import numpy as np
from torchvision import transforms
import logging
import time

from dataset import YCBVDataset
from network import SimpleModel
from utils import test, get_metrics, logger, setup_logging
from utils import matrix_to_6d_rotation


FLAGS = flags.FLAGS
flags.DEFINE_float('lr', 1e-4, 'Learning Rate')
flags.DEFINE_float('weight_decay', 1e-4, 'Weight Deacy for optimizer')
flags.DEFINE_string('output_dir', 'runs/basic/', 'Output Directory')
flags.DEFINE_string('data_dir', 'data/ycbv/v1/', 'Output Directory')
flags.DEFINE_integer('batch_size', 16, 'Batch Size')
flags.DEFINE_integer('seed', 2, 'Random seed')
flags.DEFINE_integer('max_iter', 100000, 'Total Iterations')
flags.DEFINE_integer('val_every', 1000, 'Iterations interval to validate')
flags.DEFINE_integer('save_every', 50000, 'Iterations interval to save model')
flags.DEFINE_integer('preload_images', 1, 
    'Weather to preload train and val images at beginning of training.')
flags.DEFINE_multi_integer('lr_step', [60000, 80000], 'Iterations to reduce learning rate')


log_every = 20


def main(_):
    setup_logging()
    torch.set_num_threads(4)
    torch.manual_seed(FLAGS.seed)
    # set_seed(FLAGS.seed)
    
    transform = transforms.Compose([transforms.ToTensor(), 
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                         std=[0.229, 0.224, 0.225]),])
    dataset_train = YCBVDataset(split='train', transform=transform,
                                data_dir=FLAGS.data_dir, 
                                preload_images=FLAGS.preload_images)
    dataset_val = YCBVDataset(split='val', transform=transform, 
                              data_dir=FLAGS.data_dir, 
                              preload_images=FLAGS.preload_images)
    dataset_test = YCBVDataset(split='test', transform=transform,
                               data_dir=FLAGS.data_dir,
                               preload_images=FLAGS.preload_images)
    dataloader_train = DataLoader(dataset_train, batch_size=FLAGS.batch_size,
                                  num_workers=2, shuffle=True, drop_last=True)
    
    num_classes = dataset_train.num_classes
    device = torch.device('mps') #Need to change device, because no cuda on mac, changing
    model = SimpleModel(num_classes=num_classes)
    model.to(device)

    writer = SummaryWriter(FLAGS.output_dir, max_queue=1000, flush_secs=120)
    optimizer = torch.optim.AdamW(model.parameters(), lr=FLAGS.lr, 
                                  weight_decay=FLAGS.weight_decay)
    
    milestones = [int(x) for x in FLAGS.lr_step]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=0.1)
    
    previous_best_val_metric = -1
    
    optimizer.zero_grad()
    dataloader_iter = None
    
    times_np, cls_loss_np, R_loss_np, t_loss_np, total_loss_np = [], [], [], [], []
    metrics_np = {
        'cls_accuracy': [],
        'cls_R_accuracy': [],
        'cls_t_accuracy': [],
        'cls_R_t_accuracy': [],
        'overall': [],
    }
     
    for i in range(FLAGS.max_iter):
        iter_start_time = time.time()
        
        if dataloader_iter is None or i % len(dataloader_iter) == 0:
            dataloader_iter = iter(dataloader_train)
        image, bbox, cls_gt, R_gt, t_gt, key_name = next(dataloader_iter) 
        
        image = image.to(device, non_blocking=True)
        bbox = bbox.to(device, non_blocking=True)
        cls_gt = cls_gt.to(device, non_blocking=True)
        R_gt = R_gt.to(device, non_blocking=True)
        t_gt = t_gt.to(device, non_blocking=True)

        # logits, R, t = model(image, bbox)
        batch_size = image.shape[0]

        r_curr = torch.zeros(batch_size, 6, device = device)
        t_curr = torch.zeros(batch_size, 3, device = device)
        rot_6d_gt = matrix_to_6d_rotation(R_gt).reshape(-1, 6)
        t_gt_flat = t_gt.reshape(-1,3)

        R_loss = 0
        t_loss_acc = 0

        for T in range(3):  
            logits, rot_6d, t = model(image, bbox, r_curr, t_curr)
            R_loss += nn.MSELoss()(rot_6d[torch.arange(batch_size), cls_gt], rot_6d_gt)
            t_loss_acc += nn.MSELoss()(t[torch.arange(batch_size), cls_gt], t_gt_flat)
            r_curr =rot_6d[torch.arange(batch_size), cls_gt].detach()
            t_curr = t[torch.arange(batch_size), cls_gt].detach()
            
        
        # Compute metrics
        cls_pred, R_pred, t_pred = model.process_output((logits, rot_6d, t))
        metrics = get_metrics(
            cls=cls_pred, R=R_pred, t=t_pred, 
            gt_cls=cls_gt, gt_R=R_gt, gt_t=t_gt)        
        for key, value in metrics.items():
            metrics_np[key].append(value)

        # Loss functions for training (modified since loss averaged out on T = 3 trials of IAE)
        classification_loss = nn.CrossEntropyLoss()(logits, cls_gt)
        R_loss = R_loss/3
        t_loss = t_loss_acc/3

        total_loss = classification_loss + R_loss + t_loss
        
                
        if np.isnan(total_loss.item()):
            logging.error(f'Loss went to NaN at iteration {i+1}')
            break
        
        if np.isinf(total_loss.item()):
            logging.error(f'Loss went to Inf at iteration {i+1}')
            break
        
        total_loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        # Some logging
        lr = scheduler.get_last_lr()[0]
        total_loss_np.append(total_loss.item())
        cls_loss_np.append(classification_loss.item())
        R_loss_np.append(R_loss.item())
        t_loss_np.append(t_loss.item())
        times_np.append(time.time() - iter_start_time)
                      
        if (i+1) % log_every == 0:
            print('')
            writer.add_scalar('iteration_rate', len(times_np) / np.sum(times_np), i+1)
            logger('iteration_rate', len(times_np) / np.sum(times_np), i+1)
            writer.add_scalar('loss/R', np.mean(R_loss_np), i+1)
            logger('loss/R', np.mean(R_loss_np), i+1)
            writer.add_scalar('loss/t', np.mean(t_loss_np), i+1)
            logger('loss/t', np.mean(t_loss_np), i+1)
            writer.add_scalar('lr', lr, i+1)
            logger('lr', lr, i+1)
            writer.add_scalar('loss/cls', np.mean(cls_loss_np), i+1)
            logger('loss/cls', np.mean(cls_loss_np), i+1)
            writer.add_scalar('loss/total', np.mean(total_loss_np), i+1)
            logger('loss/total', np.mean(total_loss_np), i+1)

            for key, value in metrics_np.items():
                writer.add_scalar(f'metrics/{key}', np.mean(value), i+1)
                logger(f'metrics/{key}', np.mean(value), i+1)

            times_np, cls_loss_np, R_loss_np, t_loss_np, total_loss_np = [], [], [], [], []
            metrics_np = {
                'cls_accuracy': [], 'cls_R_accuracy': [], 'cls_t_accuracy': [],
                'cls_R_t_accuracy': [], 'overall': [],
            }

        if (i+1) % FLAGS.save_every == 0:
            torch.save(model.state_dict(), f'{FLAGS.output_dir}/model_{i+1}.pth')
            
        if (i+1) % FLAGS.val_every == 0 or (i+1) == FLAGS.max_iter:
            print('')
            logging.info(f'Validating at {i+1} iterations.')
            val_dataloader = DataLoader(dataset_val, batch_size=1, num_workers=0)
            result_file_name = f'{FLAGS.output_dir}/predictions_{i+1:06d}_val.json'
            model.eval()

            results, metrics_np_val = test(val_dataloader, device, model, 
                     result_file_name)
            for key, value in metrics_np_val.items():
                writer.add_scalar(f'metrics/val-{key}', np.mean(value), i+1)
                logger(f'metrics/val-{key}', np.mean(value), i+1)
            val_metric = np.mean(metrics_np_val['overall'])
            
            if val_metric > previous_best_val_metric:
                print('')
                logging.info(f'Best validation metric improved from {previous_best_val_metric} to {val_metric}. Saving predictions on test set.')
                test_dataloader = DataLoader(dataset_test, num_workers=0,
                                 shuffle=False, drop_last=False)
                result_file_name = f'{FLAGS.output_dir}/predictions_{i+1:06d}_test.json'
                model.eval()
                test(test_dataloader, device, model, result_file_name)
                previous_best_val_metric = val_metric
                
            model.train()

    torch.save(model.state_dict(), f'{FLAGS.output_dir}/model_final.pth')

    # Save prediction result on test set
    test_dataloader = DataLoader(dataset_test, num_workers=0,
                                 shuffle=False, drop_last=False)
    result_file_name = f'{FLAGS.output_dir}/predictions_{i+1:06d}_test.json'
    model.eval()
    test(test_dataloader, device, model, result_file_name)

if __name__ == '__main__':
    app.run(main)
