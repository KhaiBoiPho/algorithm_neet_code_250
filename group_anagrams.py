from typing import List
from collections import defaultdict

class Solution:
    # My coding
    def groupAnagrams(strs: List[str]) -> List[List[str]]:
        dict_after = {}

        for word in strs:
            char_count = {}
            for char in word:
                char_count[char] = char_count.get(char, 0) + 1
            dict_after[word] = char_count

        groups = defaultdict(list)
        print(char_count)

        for word, char_count in dict_after.items():
            key = tuple(sorted(char_count.items()))
            groups[key].append(word)

        return list(groups.values())

    # Shorter coding
    def groupAnagrams1(strs: List[str]) -> List[List[str]]:
        res = defaultdict(list)
        
        for word in strs:
            count_dict = {}
            for char in word:
                count_dict[char] = count_dict.get(char, 0) + 1
            
            key = tuple(sorted(count_dict.keys()))
            res[key].append(word)
        return list(res.values())
    
    # Sorting
    def groupAnagrams2(strs: List[str]) -> List[List[str]]:
        res = defaultdict(list)
        
        for word in strs:
            sortedS = ''.join(sorted(word))
            res[sortedS].append(word)
        return list(res.values())
    
    # Hash table
    def groupAnagrams3(strs: List[str]) -> List[List[str]]:
        res = defaultdict(list)
        for word in strs:
            count = [0] * 26
            for char in word:
                count[ord(char) - ord('a')] += 1
            res[tuple(count)].append(word)
        return list(res.values())

strs = ["act","pots","tops","cat","stop","hat"]
print(Solution.groupAnagrams1(strs=strs))