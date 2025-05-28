from typing import List

class Solution:
    # res = 2,3,1,3,10,#wesay:yes!@#$%^&*()
    # My coding
    def encode(strs: List[str]) -> str:
        sizes = []
        res = ''
        
        for word in strs:
            sizes.append(len(word))
        for size in sizes:
            res += str(size)
            res += ','
        res += '#'
        for word in strs:
            res += word
        return res
        
    def decode(s: str) -> List[str]:
        res, sizes, i = [], [], 0
        
        while s[i] != '#':
            curr = ''
            while s[i] != ',':
                curr += s[i]
                i += 1
            sizes.append(int(curr))
            i += 1
        i += 1
        
        for size in sizes:
            res.append(s[i:i+size])
            i += size
        return res
    
    # Encoding & Decoding (Optimal)
    def encode1(strs: List[str]) -> str:
        res = ""
        for s in strs:
            res += str(len(s)) + "#" + s
        return res
        
    def decode1(s: str) -> List[str]:
        res = []
        i = 0
        
        while i < len(s):
            cur = i
            while cur != "#":
                cur += 1
            length = s[i:cur]
            i = cur + 1
            j = i + length
            res.append(s[i:j])
            i = j
        
        return res


strs = ["we","say",":","yes","!@#$%^&*()"]
string = Solution.encode(strs)
print(Solution.encode1(strs))
print(Solution.decode(string))