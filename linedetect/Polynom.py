import numpy as np 
from matplotlib import pyplot as plt



def main():
    coeff1 = [2.0,1.0,1.0]
    poly1 = np.poly1d(coeff1)

    coeff2 = [2.0,10.5,10.0]
    poly2 = np.poly1d(coeff2)
    t = np.linspace (-1,1,1000)

    y1 =  poly1(t)
    y2 =  poly2(t-0.5)


    Dpoly1 = poly1.deriv()
    Dpoly2 = poly2.deriv()

    dy1 = Dpoly1(t)
    dy2 = Dpoly2(t-0.5)

    plt.figure()
    plt.subplot(211)
    plt.plot(t,y1)
    plt.plot(t,y2)
    plt.subplot(212)
    plt.plot(t,dy1)
    plt.plot(t,dy2)
    plt.show()




if __name__ ==  "__main__":
    main()