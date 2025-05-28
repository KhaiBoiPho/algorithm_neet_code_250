class Solution(object):
    # Two Pointer
    def maxProfit(prices):
        left, right = 0, 1
        maxPrice = 0
        
        while right < len(prices):
            if prices[left] <= prices[right]:
                maxPrice = max(maxPrice, prices[right] - prices[left])
                right += 1
            else:
                left = right
                right += 1
        return maxPrice
    
    # Dynamic Programming
    def maxProfit2(prices):
        maxPrice = 0
        minBuy = prices[0]
        
        for sell in prices:
            maxPrice = max(maxPrice, sell - minBuy)
            minBuy = min(minBuy, sell)
        return maxPrice
    
    # Code again
    def maxProfit2(prices):
        left, right = 0, 1
        maxPrice = 0
        
        while right < len(prices):
            if prices[left] <= prices[right]:
                maxPrice = max(maxPrice, prices[right] - prices[left])
                right += 1
            else:
                left = right
                right += 1
        return maxPrice

prices = [10,1,5,6,7,1]
print(Solution.maxProfit(prices=prices))