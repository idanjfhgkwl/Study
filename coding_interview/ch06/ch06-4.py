# ## 04. 가장 흔한 단어
# Q. 금지된 단어를 제외한 가장 흔하게 등장하는 단어를 출력하라. 대소문자 구분을 하지 않으며, 구두점(마침표, 쉼표 등) 또한 무시한다.
#
# **- 입력**
# paragraph = "Bob hit a ball, the hit BALL flew far after it was hit."
# banned = ["hit"]
#
#
# ### 풀이 1. 리스트 컴프리헨션, Counter 객체 사용
# 입력값에 대소문자가 섞여 있고 쉼표 등의 구두점이 존재하므로 전처리 작업이 필요(Data Cleansing)
#
# - 정규식 사용 코드
#   - \w: 단어 문자(Word Character)
#   = ^: not

# In[7]:


import collections
import re
from typing import List


class Solution:
    def mostCommonWord(self, paragraph: str, banned: List[str]) -> str:
        words = [word for word in re.sub(r'[^\w]', ' ', paragraph)
            .lower().split()  # 소문자와 ' ' 기준으로 쪼개기
                 if word not in banned]  # banned를 제외한 단어 저장(예제에서는 hit)
        print('단어 처리: ', words)

        counts = collections.Counter(words)

        # 가장 흔하게 등장하는 단어의 첫 번째 인덱스 리턴
        # (1)은 n을 의미하며, 2차원이므로 [0][0]을 이용
        return counts.most_common(1)[0][0]


if __name__ == '__main__':
    s = Solution()
    print('\n', s.mostCommonWord(paragraph="Bob hit a ball, the hit BALL flew far after it was hit.",
                                 banned=["hit"]))