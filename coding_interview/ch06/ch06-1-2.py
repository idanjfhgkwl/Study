# ## 01. 유효한 팰린드롬
#
# Q. 주어진 문자열이 팰린드롬인지 확인하라. 대소문자를 구분하지 않으며, 영문자와 숫자만을 대상으로 한다.
# - 팰린드롬(Palindrome): 앞뒤가 똑같은 단어나 문장으로, 뒤집어도 같은 말이 되는 단어 또는 문장
#
#
# **- 입력**
# 예제 1: "A man, a plan, a canal: Panama"
# 예제 2: "race a car"
#
#
# ### 풀이 2. 데크 자료형을 이용한 최적화
# 데크(deque)를 명시적으로 선언하여 풀이 속도 개선하기
# - deque: double-ended queue; 양방향에서 데이터를 처리할 수 있는 queue형 자료구조

# In[2]:


import collections
from typing import Deque


class Solution:
    def isPalindrome(self, s: str) -> bool:

        # 자료형 데크로 선언
        strs: Deque = collections.deque()  # 데크 생성
        print('\n데크 생성: ', strs)

        for char in s:
            if char.isalnum():
                strs.append(char.lower())
                print('문자 처리: ', strs)

        while len(strs) > 1:
            if strs.popleft() != strs.pop():  # 데크의 popleft()는 O(1), 리스트의 pop(0)이 O(n)
                return False

        return True


if __name__ == '__main__':
    s = Solution()
    print(s.isPalindrome("A man, a plan, a canal: Panama"))
    print(s.isPalindrome("race a car"))