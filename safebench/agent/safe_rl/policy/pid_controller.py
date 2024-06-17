import torch
from torch.nn.functional import relu


class LagrangianPIDController:
    '''
    Lagrangian multiplier controller
    '''
    def __init__(self, KP, KI, KD, thres, per_state=True) -> None:
        super().__init__()
        self.KP = KP
        self.KI = KI
        self.KD = KD
        self.thres = thres
        self.error_old = 0
        self.error_integral = 0

    def control(self, qc):
        '''
        @param qc [batch,]
        '''
        error_new = torch.mean(qc - self.thres)  # [batch]
        error_diff = relu(error_new - self.error_old)
        self.error_integral = torch.mean(relu(self.error_integral + error_new))
        self.error_old = error_new

        multiplier = relu(self.KP * relu(error_new) + self.KI * self.error_integral +
                          self.KD * error_diff)
        return torch.mean(multiplier)
