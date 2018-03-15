

# It caclulates the mew image size
def calcResize(size, rate):
    if rate == 1:
        return size
    sizeW = size[0]//rate
    sizeH = size[1]//rate

    sizeW_ = sizeW/32
    sizeH_ = sizeH/16
    print(sizeW_, sizeH_)
    sizeWR = round(sizeW_)*32
    sizeHR = round(sizeH_)*16
    return (sizeWR, sizeHR)
