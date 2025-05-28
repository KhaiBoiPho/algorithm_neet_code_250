from typing import List

class Solution(object):
    # My coding
    def twoSum1(nums: List[int], target: int) -> List[int]:
        num_dict = {}
        for i, num in enumerate(nums):
            complement = target - num
            
            if complement in num_dict:
                return [i, num_dict[complement]]
            num_dict[num] = i
    
    # Brute Force
    def twoSum2(nums: List[int], target: int) -> List[int]:
        for i in range(len(nums)):
            for j in range(i + 1, range(len(nums))):
                if nums[i] + nums[j] == target:
                    return [i, j]
        return []
    
    # Sorting
    def twoSum3(nums: List[int], target: int) -> List[int]:
        A = []
        for i, num in enumerate(nums):
            A.append([num, i])
        
        A.sort()
        start, end = 0, len(nums) - 1
        while start < end:
            cur = A[start][0] + A[end][0]
            if cur == target:
                return [min(A[start][1], A[end][1]),
                        max(A[start][1], A[end][1])]
            elif cur < target:
                start += 1
            else:
                end -= 1
        return []
    
nums = [3,4,5,6]
target = 7
print(Solution.twoSum1(nums=nums, target=target))