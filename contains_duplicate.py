from typing import List

class Solution:
    # Brute Force
    def hasDuplicate1(self, nums: List[int]) -> bool:
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                if nums[i] == nums[j]:
                    return True
        return False
    
    # Sorting
    def hasDuplicate2(self, nums: List[int]) -> bool:
        nums.sort(reverse=True)
        for i in range(len(nums) - 1):
            if nums[i] == nums[i+1]:
                return True
        return False
    
    # Hash set
    def hasDuplicate3(self, nums: List[int]) -> bool:
        num_dict = set()
        for num in nums:
            if num in num_dict:
                return True
            num_dict.add(num)
        return False
    
    # Hash set length
    def hasDuplicate4(self, nums: List[int]) -> bool:
        return len(set(nums)) < len(nums)