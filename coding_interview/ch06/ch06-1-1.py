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
# ### 풀이 1. 리스트로 변환
# 문자열을 직접 입력받아 펠린드롬 여부 판별하기

# In[1]:


class Solution:
    def isPalindrome(self, s: str) -> bool:
        strs = []
        for char in s:
            if char.isalnum():  # isalnum(): 영문자, 숫자 여부 판별하여 False, True 변환
                strs.append(char.lower())  # 모든 문자 소문자 변환하여 str에 입력
                print('문자 처리: ', strs)

        # 팰린드롬 여부 판별
        while len(strs) > 1:  # strs의 길이가 1 이상이면 반복

            # pop(0): 맨 앞의 값, pop(): 맨 뒤의 값을 가져옴
            if strs.pop(0) != strs.pop():
                return False
        return True


if __name__ == '__main__':
    print('실행합니다: main')
    # 현재 스크립트 파일이 프로그램 시작점이 맞는지 판단
    # 스크립트 파일이 메인 프로그램으로 사용될 때와 모듈로 사용될 때를 구분하기 위함

    s = Solution()
    print(s.isPalindrome("A man, a plan, a canal: Panama"))
    print(s.isPalindrome("race a car"))