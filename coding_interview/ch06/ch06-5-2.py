# ## 05. 그룹 애너그램
# Q. 문자열 배열을 받아 애너그램 단위로 그룹핑하라.
#
# **-입력**
# ["eat", "tea", "tan", "ate", "nat", "bat"]
#
#
# ### 추가. 여러 가지 정렬 방법
# 파이썬에서 시작된 고성능 정렬 알고리즘, 팀소트(Timsort) 살펴보기
#
# #### 1. sorted() 함수를 이용한 파이썬 리스트 정렬

# In[9]:


# 숫자 정렬
a = [2, 5, 1, 9, 7]
a1 = sorted(a)
print(a1)

# In[10]:


# 문자 정렬
b = 'zbdaf'
b1 = sorted(b)
print(b1)

# #### 2. join() 함수로 리스트를 문자열로 결합

# In[11]:


b = 'zbdaf'
b1 = "".join(sorted(b))
print(b1)

# #### 3. sort() 함수로 리스트 자체를 정렬
# - 제자리 정렬(In-place Sort): 입력을 출력으로 덮어 쓰기 때문에 별도의 추가 공간이 필요하지 않고 리턴값이 없음

# In[12]:


# 알파벳 순서대로 정렬하기

c = ['ccc', 'aaaa', 'd', 'bb']
c1 = sorted(c)
print(c1)

# In[13]:


# 정렬을 위한 함수로 길이를 구하는 len을 지정
# → 알파벳 순서가 아닌 길이 순서로 정렬됨

c = ['ccc', 'aaaa', 'd', 'bb']
c1 = sorted(c, key=len)
print(c1)

# In[14]:


# 함수로 첫 문자열과 마지막 문자열 순으로 정렬(두 번째 키로 마지막 문자를 보게 한 것)
# 첫 문자열: (s[0]), 마지막 문자열: (s[-1])

a = ['cde', 'cfc', 'abc']


def fn(s):
    return s[0], s[-1]


print(sorted(a, key=fn))

# #### 4. 람다를 이용하여 정렬 처리

# In[15]:


a = ['cde', 'cfc', 'abc']
sorted(a, key=lambda s: (s[0], s[-1]))