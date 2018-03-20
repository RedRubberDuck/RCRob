import cv2
import numpy as np
import operator
import scipy.optimize


        
class LineConvter:
    
    
    @staticmethod
    def TupleList2ComplexList(line):
        newline = [] 
        for point in line:
            newline.append(complex(line[0],line[1]))
        return newline
    
    @staticmethod
    def TupleList2ArrayList(line):
        return np.array(line)

    @staticmethod
    def XY2TupleList(f_x,f_y):
        newline = [] 
        for x,y in zip(f_x,f_y):
            newline.append((int(x),int(y)))
        return newline


def LineOrderCheck(polynomLine_dic,imagesize):
    lineTestPos_dic = {}
    Y = imagesize[1]/2
    for key in polynomLine_dic:
        polynomLine = polynomLine_dic[key]
        if polynomLine.polynom is None:
            continue
        X = polynomLine.polynom(Y)
        lineTestPos_dic[key]=X
    sorted_line = sorted(lineTestPos_dic.items(), key=operator.itemgetter(1))

    newPolynomLine_dic = {}
    for index in range(len(sorted_line)):
        key,pointX = sorted_line[index]
        # print(index,key)
        newPolynomLine_dic[index] = polynomLine_dic[key]
    
    return newPolynomLine_dic







            
