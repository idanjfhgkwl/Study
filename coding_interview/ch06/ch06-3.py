# ## 03. 로그 파일 재정렬
# Q. 로그를 재정렬하라. 기준은 아래와 같다.
#   1. 로그의 가장 앞 부분은 식별자다.
#   2. 문자로 구성된 로그가 숫자 로그보다 앞에 온다.
#   3. 식별자는 순서에 영향을 끼치지 않지만, 문자가 동일한 경우 식별자 순으로 한다.
#   4. 숫자 로그는 입력 순서대로 한다.
#
#
# **- 입력**
# logs = ["dig1 8 1 5 1","let1 art can","dig2 3 6","let2 own kit dig","let3 art zero"]
#
#
# ### 풀이 1. 람다와 + 연산자 이용
# 요구 조건을 얼마나 깔끔하게 처리할 수 있는지를 묻는 문제로, 실무에서도 자주 쓰이는 로직
#
# - 조건 2, 문자로 구성된 로그가 숫자 로그 전에 오며, 숫자 로그는 입력 순서대로 둠
#   - 문자와 숫자 구분, 숫자는 그대로 이어 붙임
#   - isdigit()을 이용하여 숫자 여부를 판별해 구분해야 함

# In[6]:


from typing import List


class Solution:
    def reorderLogFiles(self, logs: List[str]) -> List[str]:
        letters, digits = [], []  # 문자, 숫자 구분

        for log in logs:
            if log.split()[1].isdigit():  # 숫자로 변환 가능한지 확인
                digits.append(log)  # 변환되면 digits에 추가
            else:  # 변환되지 않으면 letters에 추가
                letters.append(log)

        # 두 개의 키를 람다 표현식으로 정렬
        # 식별자를 제외한 문자열 [1:]을 키로 정렬하며 동일한 경우 후순위로 식별자 [0]을 지정해 정렬되도록 람자 표현식으로 정렬
        letters.sort(key=lambda x: (x.split()[1:], x.split()[0]))

        # 문자 + 숫자 순서로 이어 붙이고 return
        return letters + digits


if __name__ == '__main__':
    s = Solution()
    print(s.reorderLogFiles(["dig1 8 1 5 1", "let1 art can", "dig2 3 6", "let2 own kit dig", "let3 art zero"]))