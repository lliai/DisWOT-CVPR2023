## Teacher-free Feature Distillation

### Self-Regulated Feature Learning via Teacher-free Feature Distillation

[paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136860337.pdf), [code](https://github.com/lilujunai/Teacher-free-Distillation), [Training logs & model](https://pan.baidu.com/s/1F3QSX6MicA5qG5fxMOaCEg)(tffd), [Poster](https://github.com/lilujunai/Teacher-free-Distillation/blob/gh-pages/03287-Poster.pdf), [video](https://github.com/lilujunai/Teacher-free-Distillation/blob/gh-pages/03287.mp4), 

![03287-Poster](03287-Poster.jpg)


## Core Code
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TfFD(nn.Module):
  '''
  Teacher-free Feature Distillation
  '''
  def __init__(self, lambda_intra, lambda_inter):
    super(TfFD, self).__init__()
    self.lambda_intra = lambda_intra
    self.lambda_inter = lambda_inter
    
  def forward(self, f1, f2, f3):
    loss = (intra_fd(f1)+intra_fd(f2)+intra_fd(f3))*self.lambda_intra
    loss += (inter_fd(f1,f2)+inter_fd(f2,f3)+inter_fd(f1,f3))*self.lambda_intra
    
  def intra_fd(f_s):
    sorted_s, indices_s = torch.sort(F.normalize(f_s, p=2, dim=(2,3)).mean([0, 2, 3]), dim=0, descending=True)
    f_s = torch.index_select(f_s, 1, indices_s)
    intra_fd_loss = F.mse_loss(f_s[:, 0:f_s.shape[1]//2, :, :], f_s[:, f_s.shape[1]//2: f_s.shape[1], :, :])
    return intra_fd_loss
    
  def inter_fd(f_s, f_t):
    s_C, t_C, s_H, t_H = f_s.shape[1], f_t.shape[1], f_s.shape[2], f_t.shape[2]
    if s_H > t_H:
      f_s = F.adaptive_avg_pool2d(f_s, (t_H, t_H))
    elif s_H < t_H:
      f_t = F.adaptive_avg_pool2d(f_t, (s_H, s_H))
    else:
      pass
    inter_fd_loss = F.mse_loss(f_s[:, 0:min(s_C,t_C), :, :], f_t[:, 0:min(s_C,t_C), :, :].detach())
    return inter_fd_loss 

  return loss
```


### Bibtex 


```markdown
@inproceedings{li2022TfFD,
    title={Self-Regulated Feature Learning via Teacher-free Feature Distillation},
    author={Lujun, Li},
    booktitle={European Conference on Computer Vision (ECCV)},
    year={2022}
}
```


### Support or Contact

lilujunai@gmail.com
