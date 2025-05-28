class Solution:
    def lengthOfLongestSubstring(s: str):
        charSet = set()
        left = 0
        res = 0
        
        for right in range(len(s)):
            while s[right] in charSet:
                charSet.remove(s[left])
                left += 1
            charSet.add(s[right])
            res = max(res, len(charSet))
        return res

    # code again
    def lengthOfLongestSubstring1(s: str):
        charSet = set()
        left = 0
        res = 0
        
        for right in range(len(s)):
            while s[right] in charSet:
                charSet.remove(s[left])
                left += 1
            charSet.add(s[right])
            res = max(res, right - left + 1)
        return res
    
    # code again 2
    def lengthOfLongestSubstring1(s: str):
        charSet = set()
        left = 0
        res = 0
        
        for right in range(len(s)):
            while s[right] in charSet:
                charSet.remove(s[left])
                left += 1
            charSet.add(s[right])
            res = max(res, right - left + 1)
        return res

s="abcabcbb"
Solution.lengthOfLongestSubstring(s)