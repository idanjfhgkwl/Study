# ## 02. 문자열 뒤집기
# Q. 문자열을 뒤집는 함수를 작성하라. 입력값은 문자 배열이며, 리턴 없이 리스트 내부를 직접 조작하라.
#
# **-입력**
# 예제 1: ["h", "e", "l", "l", "o"]
# 예제 2: ["H", "a", "n", "n", "a", "H"]
#
# ### 풀이 2. 파이썬다운 방식
# 파이썬의 기본 기능을 이용하면 한 줄 코드로 불 수 있음
#
# - reverse() 함수: 리스트에만 제공되어, 문자열의 경우에는 문자열 슬라이싱으로 풀이

# In[5]:


from typing import List


class Solution:
    def reverseString(self, s: List[str]) -> None:
        s.reverse()  # 리버스는 값을 반환해주지 않고 단순히 해당 list를 뒤섞음, None 반환
        return s  # None 반환 대신 값 반환을 위해 사용


if __name__ == '__main__':
    s = Solution()
    print(s.reverseString(["h", "e", "l", "l", "o"]))
    print(s.reverseString(["H", "a", "n", "n", "a", "H"]))