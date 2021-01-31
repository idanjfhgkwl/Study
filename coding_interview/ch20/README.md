## Slinding Window 개념
- 고정 사이즈의 윈도우가 이동하면서 윈도우 내에 있는 데이터를 이용해 문제를 풀이하는 알고리즘
- 투 포인터와 vs 슬라이딩 윈도우
- 알고리즘에 대한 구체적인 설명은 아래 영상을 참고하기를 바란다. 


```python
from IPython.display import HTML
HTML('<iframe width="560" height="315" src="https://www.youtube.com/embed/0l2nePjDFuA" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>')
```




<iframe width="560" height="315" src="https://www.youtube.com/embed/0l2nePjDFuA" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>




```python
HTML('<iframe width="560" height="315" src="https://www.youtube.com/embed/iSjvuixMPmQ" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>')
```




<iframe width="560" height="315" src="https://www.youtube.com/embed/iSjvuixMPmQ" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>




```python
HTML('<iframe width="560" height="315" src="https://www.youtube.com/embed/VM1kmLrrN4Y" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>')
```




<iframe width="560" height="315" src="https://www.youtube.com/embed/VM1kmLrrN4Y" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>




```python

HTML('<iframe width="560" height="315" src="https://www.youtube.com/embed/uH9VJRIpIDY" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>')
```




<iframe width="560" height="315" src="https://www.youtube.com/embed/uH9VJRIpIDY" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>



- 두 알고리즘의 차이점에 대한 설명은 아래와 같이 정리를 할 수 있다. 

| 이름 | 정렬 여부 | 윈도우 사이즈 | 이동   |
| ------ | ------ | ------ | ------ |
| 투 포인터 | 대부분 O | 가변 | 좌우 포인터 양방향 |
| 슬라이딩 윈도우 | X | 고정 | 좌 또는 단방향 |

## Sliding-Window-Maxinum
- 문제, 배열 Num가 주어졌을 때 K 크기의 슬라이딩 윈도우를 오른쪽 끝까지 이동하면서 최대 슬라이딩 윈도우를 구하라.
- 입력 
```python
nums = [1, 3, -1, -3, 5, 3, 6, 7], k = 3
```
- 출력
```python
[3,3,5,5,6,7]
```



## 시간측정 클래스 구현
- 알고리즘 연산 속도를 측정하기 위해 아래와 같이 코드를 구현하였다. 


```python
import time

class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""

class Timer:
    def __init__(self):
        self._start_time = None

    def start(self):
        """Start a new timer"""
        if self._start_time is not None:
            raise TimerError(f"Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()

    def stop(self):
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")

        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None
        print(f"Elapsed time: {elapsed_time:0.4f} seconds")
```

### 방법 1. 브루트 포스로 계산
- 문제를 잘 읽어보면 파이썬에서는 슬라이싱과 내장 함수를 사용해 매우 쉬운 방식으로 풀이할 수 있다. 


```python
from typing import List

class Solution:
  def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
  
    print("The current nums: ", nums)
    if not nums:
      return nums
  
    r = []
    for i in range(len(nums)- k + 1):
      print("The maxNum of current", i, ": ", max(nums[i:i+k]))
      print("why", max(nums[i:i+k]), "? because The nums:", nums[i:i + k])
      r.append(max(nums[i:i + k]))
  
    return r
```


```python
nums = [1, 3, -1, -3, 5, 3, 6, 7]
k = 3
solve = Solution()
t = Timer()
t.start()
print("The final answer is:", solve.maxSlidingWindow(nums, k))
t.stop()
```

    The current nums:  [1, 3, -1, -3, 5, 3, 6, 7]
    The maxNum of current 0 :  3
    why 3 ? because The nums: [1, 3, -1]
    The maxNum of current 1 :  3
    why 3 ? because The nums: [3, -1, -3]
    The maxNum of current 2 :  5
    why 5 ? because The nums: [-1, -3, 5]
    The maxNum of current 3 :  5
    why 5 ? because The nums: [-3, 5, 3]
    The maxNum of current 4 :  6
    why 6 ? because The nums: [5, 3, 6]
    The maxNum of current 5 :  7
    why 7 ? because The nums: [3, 6, 7]
    The final answer is: [3, 3, 5, 5, 6, 7]
    Elapsed time: 0.0092 seconds
    

### 방법 2. 큐를 이용한 최적화
- Max()를 계산하는 부분에서 최적화를 하도록 한다. 
- 정렬되지 않은 슬라이딩 윈도우에서 최대값을 추출하려면 어떠한 알고리즘이든 결국 한 번 이상은 봐야 하기 때문에, 최댓값 계산을 O(n)이내로 줄일 수 있는 없다. 
- 즉, 가급적 최댓값 계산을 최소화하기 위해 이전의 최댓값을 저장해뒀다가 한칸씩 이동할 때 새 값에 대해서만 더 큰 값인지 확인한다.
- 최댓값이 윈도우에서 빠지게 되는 경우에만 다시 전체를 계산하는 형태로 개선한다. 



```python
from typing import List
import collections

class Solution:
  def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
  
    print("The current nums: ", nums)
    if not nums:
      return nums
  
    r = []
    for i in range(len(nums)- k + 1):
      print("The maxNum of current", i, ": ", max(nums[i:i+k]))
      print("why", max(nums[i:i+k]), "? because The nums:", nums[i:i + k])
      r.append(max(nums[i:i + k]))
  
    return r

  def maxSlidingWindow2(self, nums: List[int], k: int) -> List[int]:
  
    print("The current nums: ", nums)
    results = []
    window = collections.deque()
    current_max = float('-inf')
    for i, v in enumerate(nums):
      print("The current i: {}, v: {}".format(i, v))
      window.append(v)
      print("check 1: the current window is:", window)
      if i < k-1:
        continue
    
      # 새로 추가된 값이 기존 최댓값보다 큰 경우 교체
      if current_max == float('-inf'):
        current_max = max(window)
        print("check 2: current_max == float('-inf'):", current_max)

      elif v > current_max:
        current_max = v
        print("check 2: v > current_max", current_max)
      results.append(current_max)

      # 최댓값이 윈도우에서 빠지면 초기화
      if current_max == window.popleft():
        current_max = float('-inf')

    return results
```

- 큐 사용이 필요할 경우, 데크를 사용을 위해 선언한다. 
```python
import collections
window = collections.deque()
```
- 최댓값이 반영된 상태가 아니라면, 현재 윈도우 전체의 최댓값을 계산해야 한다. 
- 이미 최댓값이 존재한다면 새로 추가된 값이 기존 최댓값보다 더 큰 경우에만 최댓값을 교체한다. 
  + 이 부분이 성능 개선을 위한 핵심이다. 


```python
nums = [1, 3, -1, -3, 5, 3, 6, 7]
k = 3
solve = Solution()
t = Timer()
t.start()
print("The final answer is:", solve.maxSlidingWindow2(nums, k))
t.stop()
```

    The current nums:  [1, 3, -1, -3, 5, 3, 6, 7]
    The current i: 0, v: 1
    check 1: the current window is: deque([1])
    The current i: 1, v: 3
    check 1: the current window is: deque([1, 3])
    The current i: 2, v: -1
    check 1: the current window is: deque([1, 3, -1])
    check 2: current_max == float('-inf'): 3
    The current i: 3, v: -3
    check 1: the current window is: deque([3, -1, -3])
    The current i: 4, v: 5
    check 1: the current window is: deque([-1, -3, 5])
    check 2: current_max == float('-inf'): 5
    The current i: 5, v: 3
    check 1: the current window is: deque([-3, 5, 3])
    The current i: 6, v: 6
    check 1: the current window is: deque([5, 3, 6])
    check 2: v > current_max 6
    The current i: 7, v: 7
    check 1: the current window is: deque([3, 6, 7])
    check 2: v > current_max 7
    The final answer is: [3, 3, 5, 5, 6, 7]
    Elapsed time: 0.0074 seconds
    

- 필요할 때만 전체의 최댓값을 계산하고 이외에는 새로 추가되는 값이 최대인지만을 확인하는 형태로 계산량을 획기적으로 줄였다. 
