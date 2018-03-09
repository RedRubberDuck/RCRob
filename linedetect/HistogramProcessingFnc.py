import cv2
import numpy as np




class HistogramProcessing:

    def __init__(self,inferiorRate,superiorRate,lineThinkness,xDistanceLimit,partSize):
        self.superiorRate = superiorRate
        self.inferiorRate = inferiorRate
        self.xDistanceLimit = xDistanceLimit
        
        self.kernel =  (np.ones((1,lineThinkness))/lineThinkness)[0,:]
        self.partSize  = partSize
        partArea = partSize[1]*partSize[0]

        self.inferiorLimitSize = partArea * self.inferiorRate
        self.superiorLimitSize = partArea * self.superiorRate

    def histogramMethod(self,part,yPos):
        points=[]

        #Calculating histogram
        part1 = np.array(part,dtype=np.float)

        histogram = cv2.reduce(part1,0,rtype = cv2.REDUCE_SUM)[0,:]
        #Filter the histogram
        
        histogram_f=np.convolve(histogram,self.kernel,'same')
        # histogram_f = histogram

        accumulate=0
        accumulatePos=0
        startPx = 0
        # accumulate_a=[]
        for i in range(1,self.partSize[0]):
            #The non-zero block
            if histogram_f[i]>0:
                if histogram_f[i-1]==0:
                    startPx = i
                accumulate += histogram_f[i]
                # accumulatePos += histogram_f[i]*i
            # The end of a non-zero block
            elif histogram_f[i]==0 and histogram_f[i-1]>0:
                
                if accumulate<self.superiorLimitSize and accumulate> self.inferiorLimitSize:
                    #Calculating the middlsuperiorLimitSizee of the non-zero block
                    indexP=int((startPx+i)/2)
                    #Verify the distance from the last non-zeros block
                    if (len(points)>0 and abs(points[-1].real-indexP)<self.xDistanceLimit):
                        #If the distance is smaller than the threshold, then combines it.
                        points[-1].real=(points[-1].real+indexP)/2
                        points[-1]=(indexP,points[-1][1])
                    else:
                        # Add to the list of the windowsCenters
                        point = complex(indexP,yPos)
                        points.append(point)
                accumulate=0
                # accumulatePos=0
        return points