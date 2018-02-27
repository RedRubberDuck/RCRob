from numpy import conjugate as conj
from numpy import power 
from numpy import arctan
import math

class CubicBezierCurve:
    def __init__(self,f_A=complex(0,1),f_B=complex(0,2),f_C=complex(1,2),f_D=complex(1,3)):
        self.A=f_A
        self.B=f_B
        self.C=f_C
        self.D=f_D
        self.S=[f_A,3*(f_B-f_A),3*(f_C-2*f_B+f_A),(f_D-3*f_C+3*f_B-f_A)]
        self.dS=[self.S[1],self.S[2]*2,self.S[3]*3]
        self.ddS=[self.dS[1],self.dS[2]*2]
    
    
    def S_func(self,f_t):
        return self.S[0]+self.S[1]*f_t+self.S[2]*f_t**2+self.S[3]*f_t**3
    
    def dS_func(self,f_t):
        return self.dS[0]+self.dS[1]*f_t+self.dS[2]*f_t**2
        
    def ddS_func(self,f_t):
        return self.ddS[0]+self.ddS[1]*f_t

    def K(self,f_t):
            dS=self.dS_func(f_t)
            ddS=self.ddS_func(f_t)
            denum=power(dS*conj(dS),3.0/2.0).real
            num=((dS*conj(ddS)-conj(dS)*ddS)/(-2j)).real    
            return num/denum
    def vel(self,f_t):
        dS_r=self.dS_func(f_t)
        return sqrt((dS_r*conj(dS_r)).real)

class PolynomFunc:
    def __init__(self,coeff):
        self.coeff = coeff
    def __call__(self,t):
        res = 0
        for i in range(len(self.coeff)):
            res += self.coeff[i]*np.power(t,i)
        return res

class BezierCurve:
    def __init__(self,points):
        self.order = len(points)-1 
        self.controll = points
        self.coeff = BezierCurve.getCoeff(points)     
    
    @staticmethod
    def getCoeff(points):
        n =  len(points) - 1 
        n_fac = math.factorial(n)
        coeff = []
        for j in range(0, n +1):
            coeffJ = 0 

            for i in range(0 , j+1):
                coeffJ += power(-1,i+j)*points[i]/ (math.factorial(i)*math.factorial(j-i))
            
            coeffJ *=   n_fac / math.factorial(n-j) 
            coeff . append(coeffJ)
        return coeff


if __name__ == "__main__":
    s = 0





    
    




