from collections import defaultdict
from typing import List

class Solution:
    # My coding O(n^3) Brute Force
    def threeSum(nums: List[int]) -> List[List[int]]:
        num_dict = {}
        res = []
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                num_dict[i, j] = [nums[i] + nums[j], nums[i], nums[j]]

        for key, val in num_dict.items():
            for i in range(len(nums)):
                if i == key[0]:
                    continue
                if i == key[1]:
                    continue
                if nums[i] + val[0] == 0:
                    res.append(sorted([val[1], val[2], nums[i]]))
        
        return list(map(list, set(map(tuple, res))))
    
    # Two Pointers
    def threeSum1(nums: List[int]) -> List[List[int]]:
        res = []
        nums.sort()
        
        for i in range(len(nums)):
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            
            left = i + 1
            right = len(nums) - 1
            
            while left < right:
                total = nums[i] + nums[left] + nums[right]
                
                if total > 0:
                    right -= 1
                elif total < 0:
                    left += 1
                else:
                    res.append([nums[i], nums[left], nums[right]])

                    while left < right and nums[left] == nums[left + 1]:
                        left += 1
                    while left < right and nums[right] == nums[right - 1]:
                        right -= 1
                    
                    left += 1
                    right -= 1
        return res

nums = [-1,0,1,2,-1,-4]
print(Solution.threeSum1(nums))