from typing import List

class Solution:
    def longestConsecutive(nums: List[int]) -> int:
        num_set = set(nums)
        longest = 0
        
        for num in num_set:
            if num - 1 not in num_set:
                current = num
                length = 1
                while current + 1 in num_set:
                    current += 1
                    length += 1
                longest = max(longest, length)
        return longest

    def longestConsecutive1(nums: List[int]) -> int:
        res = 0
        store = set(nums)

        for num in nums:
            streak, curr = 0, num
            while curr in store:
                streak += 1
                curr += 1
            res = max(res, streak)
        return res

    def longestConsecutive2(nums: List[int]) -> int:
        if not nums:
            return 0

        nums.sort()
        longest = 0
        cur_streak = 1

        for i in range(1, len(nums)):
            if nums[i] == nums[i - 1]:
                continue
            elif nums[i] - nums[i - 1] == 1:
                cur_streak += 1
            else:
                longest = max(cur_streak, longest)
                cur_streak = 1
        return max(cur_streak, longest)

nums = [0,3,2,5,4,6,1,1]
print(Solution.longestConsecutive2(nums))