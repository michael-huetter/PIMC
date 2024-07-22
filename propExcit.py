# constrained randomization algorithm for the PeE move

import numpy as np
import random

def count_kinks(arr):
    n = len(arr)
    return sum(1 for i in range(n) if arr[i] != arr[(i+1) % n])

def find_kinks(arr):
    n = len(arr)
    return [i for i in range(n) if arr[i] != arr[(i + 1) % n]]

def move_kink(arr, kinks):
    
    original_arr = arr[:]
    original_kinks_count = count_kinks(arr)
    
    trial_count = 0

    while trial_count < 10:  

        if not kinks:
            return False, arr
        
        kink_index = random.choice(kinks)
        direction = random.choice([0, 1])
        
        if direction == 1:  
            target_index = (kink_index + 1) % len(arr)
        else:  
            target_index = kink_index
            kink_index = (kink_index - 1) % len(arr)
        
        arr[target_index], arr[kink_index] = arr[kink_index], arr[target_index]
        
        new_kinks_count = count_kinks(arr)
        if new_kinks_count == original_kinks_count:
            return True, arr  
        
        arr = original_arr[:]
        trial_count += 1
    
    return False, arr  


def create_or_destroy_kink(arr, n_values):

    index = random.randint(0, len(arr) - 1)
    old_value = arr[index]
    new_value = random.randint(0, n_values - 1)
    while new_value == old_value:  
        new_value = random.randint(0, n_values - 1)

    arr[index] = new_value

    return arr