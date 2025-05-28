class Solution:
    def isPalindrome(s: str) -> bool:
        string = ''.join(c for c in s if c.isalnum()).lower()
        start = 0
        end = len(string) - 1
        
        while start < end:
            if string[start] != string[end]:
                return False
            start += 1
            end -= 1
        return True
    
    def useGit():
        return 1

s="0P"
print(Solution.isPalindrome(s))