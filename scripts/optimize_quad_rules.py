#!/usr/bin/env python3


import torch
from torch import nn
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt


class QRTria(nn.Module):
    def __init__(self, N0, N1, N2):
        super().__init__()
        self.N0 = N0
        self.N1 = N1
        self.N2 = N2
        self.p0 = nn.Parameter(torch.zeros((self.N0, 0)), requires_grad=True)
        self.w0 = nn.Parameter(torch.zeros((self.N0,)), requires_grad=True)
        self.p1 = nn.Parameter(torch.zeros((self.N1,)), requires_grad=True)
        self.w1 = nn.Parameter(torch.zeros((self.N1,)), requires_grad=True)
        self.p2 = nn.Parameter(torch.zeros((self.N2, 2)), requires_grad=True)
        self.w2 = nn.Parameter(torch.zeros((self.N2,)), requires_grad=True)
        nn.init.normal_(self.p0)
        nn.init.uniform_(self.w0)
        nn.init.normal_(self.p1)
        nn.init.uniform_(self.w1)
        nn.init.normal_(self.p2)
        nn.init.uniform_(self.w2)

    def normalize_weights(self):
        wtot = torch.sum(torch.abs(self.w0)) + torch.sum(torch.abs(self.w1)) + torch.sum(torch.abs(self.w2))
        self.w0 /= 2 * wtot
        self.w1 /= 2 * wtot
        self.w2 /= 2 * wtot

    def orbits(self):
        op0 = self.p0
        ow0 = torch.abs(self.w0)
        op1 = 0.5 + 0.5*torch.tanh(self.p1)
        ow1 = torch.abs(self.w1)
        op2t = 0.5 + 0.5*torch.tanh(self.p2)
        op2 = torch.empty_like(op2t)
        op2[:,0] = 1*(op2t[:,0])*(1-op2t[:,1]) + 0.5*(op2t[:,0])*(op2t[:,1])
        op2[:,1] = 1*(1-op2t[:,0])*(op2t[:,1]) + 0.5*(op2t[:,0])*(op2t[:,1])
        ow2 = torch.abs(self.w2)
        wtot = torch.sum(ow0) + 3*torch.sum(ow1) + 6*torch.sum(ow2)
        ow0 /= 2 * wtot
        ow1 /= 2 * wtot
        ow2 /= 2 * wtot
        return (op0, ow0), (op1, ow1), (op2, ow2)

    def qr(self):
        (op0, ow0), (op1, ow1), (op2, ow2) = self.orbits()
        N = self.N0 + 3*self.N1 + 6*self.N2
        p = torch.zeros((N, 2))
        w = torch.zeros((N,))
        if self.N0 > 0:
            p[0,:] = 1/3
            w[0] = ow0[0]
        if self.N1 > 0:
            p[(self.N0+0*self.N1):(self.N0+1*self.N1),0] = op1 / 2
            p[(self.N0+0*self.N1):(self.N0+1*self.N1),1] = op1 / 2
            w[(self.N0+0*self.N1):(self.N0+1*self.N1)] = ow1
            p[(self.N0+1*self.N1):(self.N0+2*self.N1),0] = op1 / 2
            p[(self.N0+1*self.N1):(self.N0+2*self.N1),1] = 1 - op1
            w[(self.N0+1*self.N1):(self.N0+2*self.N1)] = ow1
            p[(self.N0+2*self.N1):(self.N0+3*self.N1),0] = 1 - op1
            p[(self.N0+2*self.N1):(self.N0+3*self.N1),1] = op1 / 2
            w[(self.N0+2*self.N1):(self.N0+3*self.N1)] = ow1
        if self.N2 > 0:
            p[(self.N0+3*self.N1+0*self.N2):(self.N0+3*self.N1+1*self.N2),0] = 1/3 - op2[:,0]/3 + op2[:,1]/6
            p[(self.N0+3*self.N1+0*self.N2):(self.N0+3*self.N1+1*self.N2),1] = 1/3 - op2[:,0]/3 - op2[:,1]/3
            w[(self.N0+3*self.N1+0*self.N2):(self.N0+3*self.N1+1*self.N2)] = ow2
            p[(self.N0+3*self.N1+1*self.N2):(self.N0+3*self.N1+2*self.N2),0] = 1/3 + 2*op2[:,0]/3 + op2[:,1]/6
            p[(self.N0+3*self.N1+1*self.N2):(self.N0+3*self.N1+2*self.N2),1] = 1/3 - op2[:,0]/3 - op2[:,1]/3
            w[(self.N0+3*self.N1+1*self.N2):(self.N0+3*self.N1+2*self.N2)] = ow2
            p[(self.N0+3*self.N1+2*self.N2):(self.N0+3*self.N1+3*self.N2),0] = 1/3 + 2*op2[:,0]/3 + op2[:,1]/6
            p[(self.N0+3*self.N1+2*self.N2):(self.N0+3*self.N1+3*self.N2),1] = 1/3 - op2[:,0]/3 + op2[:,1]/6
            w[(self.N0+3*self.N1+2*self.N2):(self.N0+3*self.N1+3*self.N2)] = ow2
            p[(self.N0+3*self.N1+3*self.N2):(self.N0+3*self.N1+4*self.N2),0] = 1/3 - op2[:,0]/3 + op2[:,1]/6
            p[(self.N0+3*self.N1+3*self.N2):(self.N0+3*self.N1+4*self.N2),1] = 1/3 + 2*op2[:,0]/3 + op2[:,1]/6
            w[(self.N0+3*self.N1+3*self.N2):(self.N0+3*self.N1+4*self.N2)] = ow2
            p[(self.N0+3*self.N1+4*self.N2):(self.N0+3*self.N1+5*self.N2),0] = 1/3 - op2[:,0]/3 - op2[:,1]/3
            p[(self.N0+3*self.N1+4*self.N2):(self.N0+3*self.N1+5*self.N2),1] = 1/3 + 2*op2[:,0]/3 + op2[:,1]/6
            w[(self.N0+3*self.N1+4*self.N2):(self.N0+3*self.N1+5*self.N2)] = ow2
            p[(self.N0+3*self.N1+5*self.N2):(self.N0+3*self.N1+6*self.N2),0] = 1/3 - op2[:,0]/3 - op2[:,1]/3
            p[(self.N0+3*self.N1+5*self.N2):(self.N0+3*self.N1+6*self.N2),1] = 1/3 - op2[:,0]/3 + op2[:,1]/6
            w[(self.N0+3*self.N1+5*self.N2):(self.N0+3*self.N1+6*self.N2)] = ow2
        return p, w

    def plot(self):
        p, w = self.qr()
        plt.plot([0, 1, 0, 0], [0, 0, 1, 0], color='black')
        plt.plot([0, 0.5], [0, 0.5], '--', color='black')
        plt.plot([0, 1], [0.5, 0], '--', color='black')
        plt.plot([0.5, 0], [0, 1], '--', color='black')
        plt.scatter(p[:,0].detach().to('cpu').numpy(), p[:,1].detach().to('cpu').numpy(), c=w.detach().to('cpu').numpy())
        plt.colorbar()
        plt.show()

    def forward(self, F):
        p, w = self.qr()
        return torch.sum(w * F(p))


def chebyshev_loss(degree, qr):
    def cint(n):
        if n == 1:
            return 0
        else:
            return ((-1)**n+1)/(2*(1-n**2))
    def cprod(m, n):
        return (cint(m+n) + cint(abs(m-n))) / 2
    def csimp(m, n):
        if n == 0:
            return (cprod(m, 0) - cprod(m, 1)) / 2
        elif n == 1:
            return (cprod(m, 2) - cprod(m, 0)) / 8
        else:
            return -((-1)**n)/(2*(n**2-1))*cint(m) - ((-1)**(n-1))/(4*(n-1))*cprod(m, n-1) + ((-1)**(n+1))/(4*(n+1))*cprod(m, n+1)
    p, w = qr.qr()
    loss = 0
    for i in range(degree+1):
        tmp = torch.cos(i*torch.arccos(2*p[:,0]-1))
        for j in range(degree+1-i):
            fs = tmp * torch.cos(j*torch.arccos(2*p[:,1]-1))
            I = torch.sum(w * fs)
            loss += (I - csimp(i, j))**2
    return torch.sqrt(loss) / (degree*(degree+1)/2)


def find_qr_tria(N0, N1, N2, degree, learning_rate=0.05, maxiter=10000, tol=1e-5):
    loss = 2 * tol
    while loss > tol or not torch.isfinite(loss):
        qr = QRTria(N0, N1, N2)
        optim = torch.optim.LBFGS(qr.parameters(), lr=learning_rate, max_iter=maxiter, tolerance_grad=0, tolerance_change=1e-100)
        def closure():
            optim.zero_grad()
            loss = chebyshev_loss(degree, qr)
            loss.backward()
            print(f'LBFGS loss = {loss.item()}')
            if torch.isnan(loss).item() or loss < tol:
                return 0
            return loss
        optim.step(closure)
        loss = chebyshev_loss(degree, qr)
    return qr



if __name__ == '__main__':
    deg = 20
    qr = find_qr_tria(1, 7, 7, deg)
    p, w = qr.qr()
    print(p)
    print(w)
    qr.plot()
