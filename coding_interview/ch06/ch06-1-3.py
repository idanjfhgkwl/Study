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
# ### 풀이 3. 슬라이싱 사용
# 정규식으로 불필요한 문자를 필터링하고 문자열 조작을 위해 슬라이싱 사용
# re.sub('정규표현식', 대상 문자열, 치환 문자)
# - 정규표현식: 검색 패턴 지정
# - 대상 문자열: 검색 대상이 되는 문자열
# - 치환 문자: 변경하고 싶은 문자

# In[3]:


import re  # 정규표현식 불러오기


class Solution:
    def isPalindrome(self, s: str) -> bool:
        s = s.lower()
        # 정규식으로 불필요한 문자 필터링: re.sub(''정규표현식', 대상 문자열, 치환 문자)
        s = re.sub('[^a-z0-9]', '', s)  # s 중, 알파벳과 숫자가 아닌 것을 ''로 바꿔라
        print('\n문자 처리: ', s)

        return s == s[::-1]  # 슬라이싱 [::-1]: 배열 뒤집기


if __name__ == '__main__':
    s = Solution()
    print(s.isPalindrome("A man, a plan, a canal: Panama"))
    print(s.isPalindrome("race a car"))