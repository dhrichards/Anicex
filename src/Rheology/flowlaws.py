import ufl
from Rheology import caffe, rathmann, sachs
from orientationtensors import a2calc, a4calc
class Rheology:
    def __init__(self, flowlaw = 'Glen', Ecc=1.0, Eca=100.0, n=3.0, B=100.0, α=0.0125, Emin = 0.1, Emax = 10.0):
        self.n = n
        self.B = B
        self.Ecc = Ecc
        self.Eca = Eca
        self.flowlaw = flowlaw
        self.α = α
        self.Emin = Emin
        self.Emax = Emax

    def η(self, ε, f):
        a2 = a2calc(f)
        a4 = a4calc(f)

        eps = 1.e-8
        n = self.n
        if self.flowlaw == 'Rathmann':
            η, με = rathmann.orthotropic_full(ε, a2, a4, self.Ecc, self.Eca, self.α, self.n)
            return self.B**(-1/n)*η
        elif self.flowlaw == 'CAFFE':
            E = caffe.Enhancement(caffe.Deformability(a2, a4, ε), self.Emin, self.Emax)
            return self.B**(-1/n) * (ufl.tr(ufl.dot(ε, ε)) / 2 + eps)**((1 - n) / (2 * n)) * E**(-1/n)
        else:
            return self.B**(-1/n) * (ufl.tr(ufl.dot(ε, ε)) / 2 + eps)**((1 - n) / (2 * n))

    def με(self, ε, f):

        a2 = a2calc(f, True)
        a4 = a4calc(f, True)


        if self.flowlaw == 'Sachs':
            return sachs.με(ε, a2, a4, self.Ecc, self.Eca)
        
        
        elif self.flowlaw == 'Rathmann':
            η, με = rathmann.orthotropic_full(ε, a2, a4, self.Ecc, self.Eca, self.α, self.n)
            return με
        
        else:
            return ε
        
    def τhat(self, ε, f):
        a2 = a2calc(f)
        a4 = a4calc(f)

        if self.flowlaw == 'Sachs':
            return (self.Eca/(0.4*self.Eca + 0.2*self.Ecc + 0.4))*sachs.με(ε, a2, a4, self.Ecc, self.Eca)
        

        elif self.flowlaw == 'CAFFE':
            E = caffe.Enhancement(caffe.Deformability(a2, a4, ε), self.Emin, self.Emax)
            return E**(-1/self.n)*ε
        
        
        else:
            return ε
        

    