from sachs import sachs_τhat
import ufl
import caffe
import rathmann

class Rheology:
    def __init__(self, Ecc, Eca, n, B, flowlaw='Glen', α=0.0125, Emin = 0.1, Emax = 10.0):
        self.n = n
        self.B = B
        self.Ecc = Ecc
        self.Eca = Eca
        self.flowlaw = flowlaw
        self.α = α
        self.Emin = Emin
        self.Emax = Emax

    def η(self, ε):
        eps = 1.e-8
        n = self.n
        return self.B^(1/n) * (ufl.inner(ε, ε) / 2 + eps)**((1 - n) / (2 * n))

    def τ(self, ε, f):
        a2 = a2calc(f)
        a4 = a4calc(f)

        if self.flowlaw == 'Glen':
            return self.η(ε) * ε
        
        elif self.flowlaw == 'Sachs':
            return self.η(ε)*sachs_τhat(ε, a2, a4, self.Ecc, self.Eca)
        
        elif self.flowlaw == 'CAFFE':
            E = caffe.Enhancement(caffe.Deformability(a2, a4, ε), self.Emin, self.Emax)
            return self.η(ε)*E**(-1/self.n)*ε
        
        elif self.flowlaw == 'Rathmann':
            η, με = rathmann.orthotropic_full(ε, a2, a4, self.Ecc, self.Eca, self.α, self.n)
            return η*με
        

    