import math

def getStoppingSum(toPack):
    uniqueLen = len(set(toPack))
    stoppingVal = 0
    for x in range(len(toPack) - 1, len(toPack) - uniqueLen - 1, -1):
        stoppingVal += toPack[x]
    return stoppingVal


toPack = [70,60,50,33,33,33,11,7,3]
binCapacity = 100
k = 0
curkIndex = -1

I1andI2Const = 0 
I2Size = 0
I2Sum = 0
I3Sum = 0

lowerBound = 0

I2Pos = -1

index = 0

stoppingSum = getStoppingSum(toPack)
while index < len(toPack):
    if toPack[index] <= binCapacity/2:
        curkIndex = index
        index += 1
        while index < len(toPack) and toPack[index] == toPack[curkIndex]:
            curkIndex = index
            index += 1

        # Determine I1 and I2 Size
        I1Size = 0
        for x in range(0, len(toPack)):
            if toPack[x] > binCapacity - toPack[curkIndex]:
                I1Size += 1
            elif toPack[x] > binCapacity/2:
                if I2Pos == -1:
                    I2Pos = x
                I2Size += 1
                I2Sum += toPack[x]
            elif toPack[x] >= toPack[curkIndex]:
                I3Sum += toPack[x]
            else:
                break
        
        I1andI2Const = I1Size + I2Size
        break
    index += 1

smallItemTerm = (I3Sum - (I2Size * binCapacity - I2Sum))/binCapacity
lowerBound = I1andI2Const + max(0, math.ceil(smallItemTerm))

stoppingSmallTerm = (stoppingSum - (I2Size * binCapacity - I2Sum))/binCapacity
stoppingPoint = I1andI2Const + math.ceil(stoppingSmallTerm)
print("stopping point " + str(stoppingPoint))


print(toPack)
print(curkIndex)
print(I1andI2Const)
print(I2Size)
print(I2Sum)
print(I3Sum)
print(lowerBound)

startPoint = curkIndex + 1

if index == len(toPack):
    print("BIG ITEMS")
    print(len(toPack))
else:
    while startPoint < len(toPack):
        print("HI")
        curkIndex = startPoint
        k = toPack[curkIndex]
        I3Sum += k

        curkIndex += 1

        while curkIndex < len(toPack) and toPack[curkIndex] == toPack[startPoint]:
            I3Sum += k
            startPoint = curkIndex
            curkIndex += 1

            I2Max = binCapacity - k

        while I2Pos > 0 and toPack[I2Pos - 1] <= I2Max:
            I2Size += 1
            I2Sum += toPack[I2Pos - 1]
            I2Pos -= 1

        smallItemTerm = (I3Sum - (I2Size * binCapacity - I2Sum))/binCapacity
        lowerBound = I1andI2Const + max(0, math.ceil(smallItemTerm))
        
        print(toPack)
        print(curkIndex)
        print(I1andI2Const)
        print(I2Size)
        print(I2Sum)
        print(I3Sum)
        print(lowerBound)
        stoppingSmallTerm = (stoppingSum - (I2Size * binCapacity - I2Sum))/binCapacity
        stoppingPoint = I1andI2Const + math.ceil(stoppingSmallTerm)
        print("stopping point " + str(stoppingPoint))
        
        startPoint = curkIndex



