from typing import List

class Solution:
    def maxArea(heights: List[int]) -> int:
        left = 0
        right = len(heights) - 1
        max_area = 0
        
        while left < right:
            if heights[left] < heights[right]:
                curr_area = heights[left] * (right - left)
                left += 1
            else:
                curr_area = heights[right] * (right - left)
                right -= 1
            
            if max_area < curr_area:
                max_area = curr_area
        return max_area

height = [1,7,2,5,4,7,3,6]
height1 = [2,2,2]
print(Solution.maxArea(height1))