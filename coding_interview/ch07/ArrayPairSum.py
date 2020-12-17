from typing import List


class Solution:
    def arrayPairSum1(self, nums: List[int]) -> int:
        sum = 0
        pair = []
        nums.sort()

        for n in nums:
            # 앞에서 부터 오름차순으로 페어를 만들어 합 계산
            pair.append(n)
            if len(pair) == 2:
                sum += min(pair)
                pair = []

        return sum

    def arrayPairSum2(self, nums: List[int]) -> int:
        sum = 0
        nums.sort()

        for i, n in enumerate(nums):
            # 짝수 번째 값의 합 계산
            if i % 2 == 0:
                sum += n

        return sum

    def arrayPairSum3(self, nums: List[int]) -> int:
        return sum(sorted(nums)[::2])