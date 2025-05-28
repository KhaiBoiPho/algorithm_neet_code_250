from typing import List

class Solution:
    # Extra Space
    def rotate(nums: List[int], k: int) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        n = len(nums)
        tmp = [0] * n
        for i in range(n):
            tmp[(i + k) % n] = nums[i]
        nums[:] = tmp

    # One Liner
    def rotate1(nums: List[int], k: int) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        nums[:] = nums[-k % len(nums):] + nums[:-k % len(nums)]

    def rotate2(nums: List[int], k: int) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        k = k % len(nums)
        nums[:] = nums[len(nums) - k:] + nums[:len(nums) - k]