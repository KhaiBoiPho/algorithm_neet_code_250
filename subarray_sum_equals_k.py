class Solution(object):
    def subarraySum(nums, k):
        count = 0
        sum = 0
        d = {0: 1}
        
        for num in nums:
            sum += num
            if (sum - k) in d:
                count += d[sum - k]
            
            if sum in d:
                d[sum] += 1
            else:
                d[sum] = 1
        return count
    
    # code again
    def subarraySum1(nums, k):
        count = 0
        sum = 0
        d = { 0 : 1 }
        
        for num in nums:
            sum += num
            if (sum - k) in d:
                count += d[sum - k]
            
            d[sum] = d.get(sum, 0) + 1
        return count

nums = [1,2,1,1,2,3]
k = 4
print(Solution.subarraySum(nums=nums, k=k))