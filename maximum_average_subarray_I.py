class Solution(object):
    def findMaxAverage(nums, k):
        start = 0
        average = float(0)
        
        for i in range(k):
            average += nums[i]
        average = average / k
        print(average)
        
        if len(nums) == 1 or k == len(nums):
            return average
        
        max_avg = average
        for i in range(k, len(nums)):
            average = ((average * k) - nums[start] + nums[i]) / k
            if average > max_avg:
                max_avg = average
            start += 1
        return max_avg
    
    def findMaxAverage2(nums, k):
        window_sum = sum(nums[:k])
        max_sum = window_sum
        
        for i in range(k, len(nums)):
            window_sum += nums[i] - nums[i - k]
            max_sum = max(max_sum, window_sum)
        
        return float(max_sum) / k

nums = [7,4,5,8,8,3,9,8,7,6]
k = 7
print(Solution.findMaxAverage2(nums, k))