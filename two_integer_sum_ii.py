from typing import List

class Solution:
    # My coding use two pointer
    def twoSum(numbers: List[int], target: int) -> List[int]:
        start = 0
        end = len(numbers) - 1
        
        while start < end:
            if numbers[start] + numbers[end] > target:
                end -= 1
            elif numbers[start] + numbers[end] < target:
                start += 1
            else:
                return [start + 1, end + 1]
        return []
    
    # Shorter my coding
    def twoSum1(numbers: List[int], target: int) -> List[int]:
        i, j = 0, len(numbers) - 1
        while i < j:
            total = numbers[i] + numbers[j]
            if total == target: return [i + 1, j + 1]
            if total < target: i += 1
            else: j -= 1
    
    # Hash map
    def twoSum2(numbers: List[int], target: int) -> List[int]:
        num_dict = {}
        
        for index, num in enumerate(numbers):
            complement = target - num
            
            if complement in num_dict:
                return [num_dict[complement] + 1, index + 1]
            num_dict[num] = index
        return []

numbers=[1,3,4,7,8]
target=9
print(Solution.twoSum2(numbers, target))