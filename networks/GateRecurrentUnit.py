
import torch
import torch.nn as nn
import torch.nn.functional as func

class GateRecurrentUnit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3):
        super(GateRecurrentUnit, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.gate_conv = nn.Conv2d(in_channels + out_channels, out_channels * 2, kernel_size, padding=1)
        self.reset_gate_norm = nn.GroupNorm(1, out_channels, 1e-5, True)
        self.update_gate_norm = nn.GroupNorm(1, out_channels, 1e-5, True)
        self.output_conv = nn.Conv2d(in_channels + out_channels, out_channels, kernel_size, padding=1)
        self.output_norm = nn.GroupNorm(1, out_channels, 1e-5, True)

    def gates(self, x, h):
        c = torch.cat((x, h), dim=1)
        f = self.gate_conv(c)
        channels = f.shape[1]
        r, u = torch.split(f, channels // 2, 1)

        rn = self.reset_gate_norm(r)
        un = self.update_gate_norm(u)
        rns = func.sigmoid(rn)
        uns = func.sigmoid(un)
        return rns, uns

    def output(self, x, h, r):
        f = torch.cat((x, r * h), dim=1)
        o = self.output_conv(f)
        on = self.output_norm(o)
        return on

    def forward(self, x, h):
        r, u = self.gates(x, h)
        o = self.output(x, h, r)
        y = func.tanh(o)
        return u * h + (1 - u) * y
