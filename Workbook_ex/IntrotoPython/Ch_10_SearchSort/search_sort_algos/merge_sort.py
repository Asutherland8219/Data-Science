import numpy as np 

def merge_sort(data):
    sort_array(data, 0, len(data) - 1)

def sort_array(data, low, high):
    """ Split then sort """
    if (high - low) >= 1:
        middle1 = (low + high) // 2
        middle2 = middle1 + 1

        #output split step 
        print(f'split:  {subarray_string(data, low, high)}')
        print(f'        {subarray_string(data, low, middle1)}')
        print(f'        {subarray_string(data, middle2, high)}')

        # split in half, sort recursively 
        sort_array(data, low, middle1) #half 1
        sort_array(data, middle2, high) #half 2

        # merge the two sorted arrays after split calls return 
        merge(data, low, middle1, middle2, high)


def merge(data, left, middle1, middle2, right):
    left_index = left 
    right_index = middle2
    combined_index = left
    merged = [0] * len(data)

    # output two sub arrays before merging 
    print(f'merge:  {subarray_string(data, left, middle1)}')
    print(f'        {subarray_string(data, middle2, right)}')

    while left_index <= middle1 and right_index <= right:
        if data[left_index] <= data[right_index]:
            merged[combined_index] = data[left_index]
            combined_index += 1
            left_index += 1
        else:
            merged[combined_index] = data[right_index]
            combined_index += 1
            right_index += 1

    # if array is left empty 
    if left_index == middle2:
        merged[combined_index:right + 1] = data[right_index: right + 1]
    else:
        merged[combined_index: right + 1] = data[left_index: middle1 + 1]
    
    data[left:right + 1] = merged[left:right + 1]

    # output merged array 
    print(f'     {subarray_string(data, left, right)}\n')

def subarray_string(data, low, high):
    temp = '   ' * low 
    temp += ' '.join(str(item) for item in data[low:high +  1])
    return temp

def main():
    data = np.arrya([35, 73, 90, 65, 23, 86, 43, 81, 34, 58])
    print(f'Unsorted array: {data}\n')
    merge_sort(data)
    print(f'Sorted array: {data}\n')

if __name__ == '__main__':
    main()
   

