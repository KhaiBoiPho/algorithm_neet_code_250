class Solution(object):
    def sortedSquares(nums):
        left = 0
        right = len(nums) - 1
        res = [0] * len(nums)
        
        for i in range(len(nums) - 1, -1, -1):
            if abs(nums[left]) < abs(nums[right]):
                res[i] = nums[right] ** 2
                right -= 1
            else:
                res[i] = nums[left] ** 2
                left += 1
        return res


nums = [-4,-1,0,3,10]
print(Solution.sortedSquares(nums=nums))