class Solution(object):
    def removeElement(nums, val):
        del_pos = 0
        for num in nums:
            if num != val:
                nums[del_pos] = num
                del_pos += 1
        return del_pos

Solution.removeElement([0,1,2,2,3,0,4,2], 2)