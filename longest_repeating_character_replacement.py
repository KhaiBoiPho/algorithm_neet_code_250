class Solution:
    # Brute Force
    def characterReplacement(s: str, k: int) -> int:
        res = 0
        for i in range(len(s)):
            count, maxf = {}, 0
            for j in range(i, len(s)):
                count[s[j]] = count.get(s[j], 0) + 1
                maxf = max(maxf, count[s[j]])
                if (j - i + 1) - maxf <= k:
                    res = max(res, j - i + 1)
        return res

    # Sliding Window (Optimal)
    def characterReplacement1(s: str, k: int) -> int:
        count = {}
        res = 0
        
        left = 0
        maxf = 0
        for right in range(len(s)):
            count[s[right]] = count.get(s[right], 0) + 1
            maxf = max(maxf, count[right])
            
            while (right - left + 1) - maxf > k:
                count[s[left]] -= 1
                left += 1
            res = max(res, right - left + 1)
        return res

s = "AAABABBCC"
k = 2
Solution.characterReplacement(s, k)