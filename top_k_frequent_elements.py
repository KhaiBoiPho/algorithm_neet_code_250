from typing import List
from collections import defaultdict

class Solution:
    # My coding
    def topKFrequent(nums: List[int], k: int) -> List[int]:
        num_dict = defaultdict(int)
        
        for num in nums:
            num_dict[num] = num_dict.get(num, 0) + 1
        
        res = [item[0] for item in (sorted(num_dict.items(), key=lambda item: item[1], reverse=True))][:k]
        return res
    
    # Bucket sort
    def topKFrequent1(nums: List[int], k: int) -> List[int]:
        num_dict = {}
        freq = [[] for i in range(len(nums) + 1)]
        
        for num in nums:
            num_dict[num] = num_dict.get(num, 0) + 1
        for num, count in num_dict.items():
            freq[count].append(num)
        
        res = []
        for i in range(len(freq) - 1, 0, -1):
            for num in freq[i]:
                res.append(num)
                if len(res) == k:
                    return res
    
    # My coding again
    def topKFrequent2(nums: List[int], k: int) -> List[int]:
        num_dict = {}
        
        for num in nums:
            num_dict[num] = num_dict.get(num, 0) + 1
        
        res = [item[0] for item in sorted(num_dict.items(), key=lambda x: x[1], reverse=True)][:k]
        return res

    # Bucket sort again
    def topKFrequent3(nums: List[int], k: int) -> List[int]:
        num_dict = {}
        freq = [[] for i in range(len(nums) + 1)]
        
        for num in nums:
            num_dict[num] = num_dict.get(num, 0) + 1
        for num, count in num_dict.items():
            freq[count].append(num)
        
        res = []
        for i in range(len(freq) - 1, 0, -1):
            for num in freq[i]:
                res.append(num)
            if len(res) == k:
                return res

nums = [1,2,2,3,3,4,7,9,9]
k = 3
print(Solution.topKFrequent3(nums, k))