import torch
import json
import logging
import numpy as np
import torch.nn.functional as F

def setup_logging():
    log_formatter = logging.Formatter(
        '%(asctime)s: %(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    logging.getLogger().handlers = []
    if not len(logging.getLogger().handlers): 
        logging.getLogger().addHandler(console_handler)
    logging.getLogger().setLevel(logging.INFO)

def logger(tag, value, global_step):
    if tag == '':
       logging.info('')
    else:
       logging.info(f'  {tag:>30s} [{global_step:07d}]: {value:5f}')

def make_transform(R, t):
    T = torch.cat((R, t), dim=2)
    T = torch.nn.functional.pad(T, (0, 0, 0, 1))
    T[:, 3, 3] = 1
    return T

def make_rotation_matrix(R_in):
    with torch.no_grad():
        R_in = R_in.reshape(-1, 3, 3)
        U, S, Vh = torch.linalg.svd(R_in, full_matrices=True)

        S = S - S + 1
        # Check determinant sign of U@V^T
        det = torch.det(U @ Vh)
        # For those with negative det, flip one of the singular values to -1.
        neg_idx = det < 0
        # Flip the sign of the last column of U for rows where det < 0
        S[neg_idx, -1] *= -1
        R_out = U @ torch.diag_embed(S) @ Vh
        return R_out
     
def get_metrics(cls, R, t, gt_cls, gt_R, gt_t, R_thresh=0.5, t_thresh=0.1):
    cls_accuracy = torch.mean((cls == gt_cls).float())
    gt_transform = make_transform(gt_R, gt_t)
    R = make_rotation_matrix(R)
    pred_transform = make_transform(R, t)
    relative_transform = torch.linalg.inv(gt_transform) @ pred_transform
    R_error = relative_transform[:, :3, :3]
    t_error = relative_transform[:, :3, 3:]
    trace_R_error = R_error[:,0,0] + R_error[:,1,1] + R_error[:,2,2]   
    R_error_radians = torch.acos(torch.clamp((trace_R_error - 1) / 2, min=-1, max=1))
    t_error_m = torch.linalg.vector_norm(t_error, dim=1)
    cls_R_accuracy = torch.mean(
        torch.logical_and(cls == gt_cls, R_error_radians < R_thresh).float())
    cls_t_accuracy = torch.mean(
        torch.logical_and(cls == gt_cls, t_error_m < t_thresh).float())
    cls_R_t_accuracy = torch.mean(
        torch.logical_and(
            torch.logical_and(cls == gt_cls, t_error_m < t_thresh), 
            R_error_radians < R_thresh).float())
    out = {
        'cls_accuracy': cls_accuracy.item(),
        'cls_R_accuracy': cls_R_accuracy.item(),
        'cls_t_accuracy': cls_t_accuracy.item(),
        'cls_R_t_accuracy': cls_R_t_accuracy.item()
    }
    out['overall'] = np.sum(list(out.values()))
    return out

def test(dataloader, device, model, result_file_name):
    model.eval()
    results = {}
    metrics_np = {'cls_accuracy': [], 'cls_R_accuracy': [], 
                  'cls_t_accuracy': [],'cls_R_t_accuracy': [],
                  'overall': []}
    for i, item in enumerate(dataloader):
        image, bbox, gt_cls, gt_R, gt_t, key_name = item 
        image = image.to(device, non_blocking=True)
        bbox = bbox.to(device, non_blocking=True)
        gt_cls = gt_cls.to(device, non_blocking=True)
        gt_R = gt_R.to(device, non_blocking=True)
        gt_t = gt_t.to(device, non_blocking=True)
        with torch.no_grad():
            outs = model(image, bbox)
            cls, R, t = model.process_output(outs)
        
        results[key_name[0]] = {
            'cls': cls.reshape(1).detach().cpu().numpy().tolist(),
            'R': R.reshape(9).detach().cpu().numpy().tolist(),
            't': t.reshape(3).detach().cpu().numpy().tolist(),
        }
        metrics = get_metrics(cls, R, t, gt_cls=gt_cls, gt_R=gt_R, gt_t=gt_t)
        
        for key, value in metrics.items():
            metrics_np[key].append(value)
    
    # Write results to file
    json.dump(results, open(result_file_name, 'w'), indent=4)
    return results, metrics_np

def rotation_6d_to_matrix(d6 : torch.Tensor) -> torch.Tensor:
    '''
    Retrieved from http://arxiv.org/abs/1812.07035
    Read paper to get better understanding of conversion 
    '''

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim = 1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim = 1)
    b3 = torch.cross(b1, b2, dim = 1)
    return torch.stack((b1, b2, b3), dim = -2)

def matrix_to_6d_rotation(matrix: torch.Tensor) -> torch.Tensor:
    '''
    Converts to 6d representation by dropping last row 
    '''
    return matrix[..., :2, :].clone().reshape(*matrix.size()[:-2], 6)



