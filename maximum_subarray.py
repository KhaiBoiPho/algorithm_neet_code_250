class Solution(object):
    def maxSubArray(nums):
        curMax = maxTillNow = nums[0]
        
        for num in nums[1:]:
            curMax = max(num, curMax + num)
            maxTillNow = max(maxTillNow, curMax)
        return maxTillNow

nums = [-2,1,-3,4,-1,2,1,-5,4]
print(Solution.maxSubArray(nums=nums))