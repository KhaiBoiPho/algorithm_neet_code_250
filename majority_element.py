class Solution(object):
    def majorityElement1(nums):
        num_dict = {}
        max_freq = 0
        need_out = int(len(nums) / 2)
        for num in nums:
            if num in num_dict:
                num_dict[num] += 1
            else:
                num_dict[num] = 1
        print(num_dict)
        
        for num, times in num_dict.items():
            if times > max_freq:
                max_freq = times
        
        if max_freq > need_out:
            return [num for num, times in num_dict.items() if times == max_freq][0]
        else:
            return 0
    
    def majorityElement2(nums):
        num_dict = {}
        need_out = len(nums) // 2
        
        for num in nums:
            num_dict[num] = num_dict.get(num, 0) + 1
            if num_dict[num] > need_out:
                return num
        
        return 0

    def bestMajorityElement(nums):
        count = 0
        candidate = None
        for num in nums:
            if count == 0:
                candidate = num
            count += (1 if candidate == num else -1)
        return candidate


nums = [2,2,1,1,1,2,2]
print(Solution.bestMajorityElement(nums))