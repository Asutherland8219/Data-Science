import numpy as np 

def binary_search(data, key):
    "Perform binary search of data looking for key"
    low = 0 # min value
    high = len(data) - 1 # max value 
    middle = (low + high + 1) // 2
    location = -1 # return value if not found 

    # loop search for element 
    while low <= high and location == -1:
        print(remaining_elements(data, low, high))

        print('   ' * middle, end='')
        print(' * ')

        # if element is found in the middle 
        if key == data[middle]:
            location = middle
        elif key < data[middle]:
            high = middle - 1
        else:
            low = middle + 1

        middle = (low + high + 1) // 2 # recalculate middle

    return location #returns the location of search key

def remaining_elements(data, low , high):
    return '    ' * low + ' '.join(str(s) for s in data[low:high + 1])


