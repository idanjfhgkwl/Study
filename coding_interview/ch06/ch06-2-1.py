# ## 02. 문자열 뒤집기
# Q. 문자열을 뒤집는 함수를 작성하라. 입력값은 문자 배열이며, 리턴 없이 리스트 내부를 직접 조작하라.
#
# **-입력**
# 예제 1: ["h", "e", "l", "l", "o"]
# 예제 2: ["H", "a", "n", "n", "a", "H"]
#
# ### 풀이 1. 투 포인터를 이용한 스왑
# - 투 포인터(Two Pointer): 2개의 포인터를 이용해 범위를 조정해가며 풀이하는 방식
#
# 문제에 '리턴 없이 리스트 내부를 직접 조작하라'는 제약이 있으므로 s 내부를 스왑하는 형태로 풀이할 수 있음

# In[4]:


from typing import List


class Solution:
    def reverseString(self, s: List[str]) -> None:
        left, right = 0, len(s) - 1
        while left < right:
            s[left], s[right] = s[right], s[left]
            left += 1
            right -= 1
        return s


if __name__ == '__main__':
    s = Solution()
    print(s.reverseString(["h", "e", "l", "l", "o"]))
    print(s.reverseString(["H", "a", "n", "n", "a", "H"]))