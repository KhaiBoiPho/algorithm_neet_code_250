class Solution:
    # My coding
    def isAnagram(s: str, t: str) -> bool:
        char_dict_1 = {}
        char_dict_2 = {}
        for char in s:
            if char in char_dict_1:
                char_dict_1[char] += 1
            else:
                char_dict_1[char] = 1

        for char in t:
            if char in char_dict_2:
                char_dict_2[char] += 1
            else:
                char_dict_2[char] = 1
        
        return char_dict_1 == char_dict_2
    
    # Sorting
    def isAnagram1(s: str, t: str) -> bool:
        if len(s) != len(t):
            return False
        
        return sorted(s) == sorted(t)
    
    # Hash map
    def isAnagram2(s: str, t: str) -> bool:
        if len(s) != len(t):
            return False
        
        countS, countT = {}
        
        for i in range(len(s)):
            countS[s[i]] = countS.get(s[i], 1) + 1
            countT[s[i]] = countT.get(t[i], 1) + 1
        return countS == countT
    
    # Hash table (Using array)
    def isAnagram3(s: str, t: str) -> bool:
        if len(s) != len(t):
            return False
        
        count = [0] * 26
        for i in range(len(s)):
            count[ord(s[i]) - ord('a')] += 1
            count[ord(t[i]) - ord('a')] -= 1
        
        for val in count:
            if val != 0:
                return False
        return True

s="racecar"
t="carrace"
print(Solution.isAnagram(s, t))