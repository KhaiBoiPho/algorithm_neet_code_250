from typing import List

class Solution:
    # Brute Force
    def productExceptSelf(nums: List[int]) -> List[int]:
        res = []
        mul = 1
        for i in range(len(nums)):
            for j in range(len(nums)):
                if i != j:
                    mul *= nums[j]
            res.append(mul)
            mul = 1
        return res
    
    # Division
    def productExceptSelf1(nums: List[int]) -> List[int]:
        res = [0] * len(nums)
        mul = 1
        zero_count = nums.count(0)
        
        if zero_count > 1:
            return res
        
        for num in nums:
            if num != 0:
                mul *= num
        
        if zero_count == 1:
            for i in range(len(nums)):
                if nums[i] == 0:
                    res[i] = mul
            return res
        else:
            for i in range(len(nums)):
                res[i] = mul // nums[i]
            return res
    
    # Prefix & Suffix
    def productExceptSelf2(nums: List[int]) -> List[int]:
        n = len(nums)
        res = [0] * n
        pref = [0] * n
        suff = [0] * n
        
        pref[0] = suff[n - 1] = 1
        for i in range(1, n):
            pref[i] = nums[i - 1] * pref[i - 1]
        for i in range(n - 2, -1, -1):
            suff[i] = nums[i + 1] * suff[i + 1]
        for i in range(n):
            res[i] = pref[i] * suff[i]
        return res

    # Prefix & Suffix (Optimal)
    def productExceptSelf3(nums: List[int]) -> List[int]:
        res = [1] * len(nums)
        
        prefix = 1
        for i in range(len(nums)):
            res[i] = prefix
            prefix *= nums[i]
        postfix = 1
        for i in range(len(nums) - 1, -1, -1):
            res[i] *= postfix
            postfix *= nums[i]
        return res

nums = [1,2,4,6]
print(Solution.productExceptSelf2(nums))