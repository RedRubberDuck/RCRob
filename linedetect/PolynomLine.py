import numpy as np

class PolynomLine:
    
    limit_K = np.sqrt(2)/(50*4)
    
    def __init__(self,polyDeg):
        self.polyDeg = polyDeg
        self.polynom = None
        self.dPolynom = None 
        self.line = []
        self.lineInterval = None
    def estimatePolynom(self,line):
        if len(line) <= self.polyDeg:
            print("Warming: Not enough to estimate the polynom")
            return
        # l_point_a = np.array(line)
        l_y = np.imag(line)
        l_x = np.real(line)
        # print(line,l_y,l_x)

        

        # if self.polynom  is not None:
            # popt,pcov = scipy.optimize.curve_fit(PolynomLine.poly,l_y,l_x,p0=self.polynom.coef)
            # print((self.polynom.coef-popt))
        coeff1 = np.polyfit(l_y,l_x,self.polyDeg) 
            # error = abs(self.polynom.coef-coeff1)
            # if (error[0] < 0.001 and error[1] < 5 and error[2]<500):
        # print('Min-R:',abs(coeff1[0]),PolynomLine.limit_K)
        if (abs(coeff1[0]) < PolynomLine.limit_K):
            coeff = coeff1
        else:
            print("To big curvature.")
            return
            # else:
            #     print ("error:",error)
            #     coeff = self.polynom.coef
            # coeff = (coeff*0.5 + self.polynom.coef*0.9)
        # else:
        #     coeff = np.polyfit(l_y,l_x,self.polyDeg) 
        self.polynom = np.poly1d(coeff)
        self.dPolynom = self.polynom.deriv()
        self.line = line
        self.lineInterval = [np.min(l_y),np.max(l_y)]

    def poly(x,*coeff):
        return np.poly1d(coeff)(x)    
    
    def generateLinePoint(self,line):
        if (self.polynom is None):
            print("Warming: Polynom wasn't initialized!")
            return 
        l_point_a = LineConvter.TupleList2ArrayList(line)
        l_y = l_point_a[:,1]
        l_x = self.polynom(l_y)
        return LineConvter.XY2TupleList(l_x,l_y)
