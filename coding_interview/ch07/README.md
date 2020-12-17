# Chap7 배열

자료구조는 메모리 공간 기반의 연속(Contiguous) 방식과 포인터 기반의 연결(Link) 방식으로 나뉜다.  
배열은 이 중에서 연속 방식의 가장 기본이 되는 자료형이다.

배열(array)이란?
- 배열(array)은 같은 타입의 변수들로 이루어진 유한 집합으로 정의된다.
- 배열을 구성하는 각각의 값을 배열 요소(element)라고 하며, 배열에서의 위치를 가리키는 숫자는 인덱스(index)라고 한다.
- C언어에서 인덱스는 언제나 0부터 시작하며, 0을 포함한 양의 정수만을 가질 수 있다.
- C언어에서 배열은 크기를 지정하고 해당 크기만큼의 연속된 메모리 공간을 할당받는 작업을 수행하는 자료형을 말한다.
- 크기가 고정되어 있으며, 한번 생성한 배열은 크기를 변경하는 것이 불가능하다.

![array](https://user-images.githubusercontent.com/72365663/102178542-f2192100-3ee8-11eb-9aee-c8a5a2fdba3e.JPG)

- 예를 들어 성적 배열을 입력할 경우
```
int grade[3] = {85, 65, 90};    #길이가 3인 int형 배열 선언  
```
    - 다음 그림은 배열 grade가 메모리 상에서 어떻게 저장되는지를 보여준다.
![img_c_onedimensional_array](https://user-images.githubusercontent.com/72365663/102178766-56d47b80-3ee9-11eb-8012-f5e2926289d2.png)
    - int의 경우 32비트 이상의 시스템에서는 4바이트를 사용한다. 따라서 각 element는 4바이트, 배열의 주소 또한 4씩 증가한다.


앞서 배열이란 고정된 크기만큼의 연속된 메모리 할당이라고 설명했다. 하지만 실제 데이터에서는 전체의 크기를 가늠하기 힘들어서 크기를 할당하는데 어려움이 있다.  
이를 해결하고자 크기를 지정하지 않고 자동으로 리사이징하는 배열인 **동적 배열**이 등장했다.
 - 자바에서는 ArrayList, C++에서는 std::vector, 파이썬에서는 리스트가 대표적인 동적 배열 자료형이다.
 - 대부분의 동적 프로그래밍 언어들은 아예 정적 배열 자체가 없다. 즉 파이썬에서도 정적 배열은 따로 제공하지 않으며 동적 배열인 리스트만 제공한다.
 - 동적 배열의 원리는 미리 초깃값을 작게 잡아 배열을 생성하고, 데이터가 추가되면서 꽉 채워지면, 늘려주고 모두 복사하는 식이다.

## Q7 두 수의 합

덧셈하여 타겟을 만들 수 있는 배열의 두 숫자 인덱스를 리턴하라.

- 입력
```
nums = [2, 7, 11, 15], target = 9
```
- 출력
```
[0, 1]
```
- 설명
    - num[0] + num[1] = 2 + 7 = 9
    - 따라서 0, 1을 리턴한다.

### Solution 1 브루트 포스로 계산

브루스 포스(Brute - Force)은 모든 조합을 더해서 일일이 확인해보는 무차별 대입 방식이다.  
![Bruthe-Force](https://user-images.githubusercontent.com/72365663/102181339-b46ac700-3eed-11eb-9b21-81e12a76bf50.JPG)


```python
from typing import List


class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:    # twoSum이라는 함수 정의(nums라는 리스트입력, 타겟값입력)
        for i in range(len(nums)):    # i는 0부터 리스트의 길이(문제에서는 4)까지 범위의 값들을 반복수행
            for j in range(i + 1, len(nums)):   # j는 i+1부터 리스트의 길이(문제에서는 4)까지 범위의 값들을 반복수행
                if nums[i] + nums[j] == target:   # index 값이 i와 j인 값을 더해서 target 값과 같은지 비교
                    return [i, j]   # 같을 경우 해당 index값인 i, j 값을 반환

if __name__ == '__main__':
  s = Solution()
  print(s.twoSum([2, 7, 11, 15], 9))
```

    [0, 1]
    

이 풀이는 5,248밀리초나 소요됐다.  
브루트 포스 방식은 무차별 대입으로 인해 상당히 비효율적이며 지나치게 느리다.  
따라서 좀 더 최적화할 수 있는 방안을 고민해야 한다.

### Solution 2 in을 이용한 탐색

모든 조합을 비교하지 않고 정답, 즉 타겟에서 첫 번째 값을 뺀 값 target - n이 존재하는지 탐색하는 문제로 변경해보자.

![in](https://user-images.githubusercontent.com/72365663/102222469-ccab0800-3f26-11eb-853d-ef8b4f5615c2.JPG)


```python
from typing import List


class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        for i, n in enumerate(nums):    # i는 인덱스 값을, n은 해당 인덱스 값에 해당하는 value 값을 불러온다
            complement = target - n   # 타겟값에 n값을 뺀 새로운 수를 정의
            print(complement)

if __name__ == '__main__':
  s = Solution()
  s.twoSum([2, 7, 11, 15], 9)
```

    7
    2
    -2
    -6
    


```python
from typing import List


class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        for i, n in enumerate(nums):    # i는 인덱스 값을, n은 해당 인덱스 값에 해당하는 value 값을 불러온다
            complement = target - n   # 타겟값에 n값을 뺀 새로운 수를 정의

            if complement in nums[i + 1:]:    #만약 타겟값에서 n값을 뺀 값이 i보다 큰 인덱스의 value값에 존재할 경우 그 값을 불러온다
                return [nums.index(n), nums[i + 1:].index(complement) + (i + 1)]

if __name__ == '__main__':
  s = Solution()
  print(s.twoSum([2, 7, 11, 15], 9))
```

    [0, 1]
    

앞의 solution과 같은 시간 복잡도라도 in 연산 쪽이 훨씬 더 가볍고 빠르다.  
코드도 864밀리초 만에 실행이 가능하다.

### Solution 3 첫 번째 수를 뺀 결과 키 조회

이번에는 시간 복잡도를 개선해 속도를 높여보자.  
이번 솔루션에서는 타겟에서 첫 번째 수를 빼면 두 번째 수를 바로 알아낼 수 있는 점을 이용할 것이다.  
두 번째 수를 키로 하고 기존의 인덱스는 값으로 바꿔서 딕셔너리로 저장해두면, 나중에 두 번째 수를 키로 조회해서 정답을 즉시 찾아낼 수 있을 것이다.

![search_key](https://user-images.githubusercontent.com/72365663/102232349-57453480-3f32-11eb-98fd-ea4603bd120f.JPG)


```python
from typing import List


class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        nums_map = {}
        # 키와 값을 바꿔서 딕셔너리로 저장
        for i, num in enumerate(nums):
            nums_map[num] = i
        print(nums_map)

if __name__ == '__main__':
  s = Solution()
  s.twoSum([2, 7, 11, 15], 9)   # value값은 key로, index는 value로 바꿔서 dictionary로 저장
```

    {2: 0, 7: 1, 11: 2, 15: 3}
    


```python
from typing import List


class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        nums_map = {}
        # 키와 값을 바꿔서 딕셔너리로 저장
        for i, num in enumerate(nums):
            nums_map[num] = i

        # 타겟에서 첫 번째 수를 뺀 결과를 키로 조회
        for i, num in enumerate(nums):
            if target - num in nums_map and i != nums_map[target - num]:  # target - num 값이 nums_map에 존재하고, 
                print(nums_map[target - num])

if __name__ == '__main__':
  s = Solution()
  s.twoSum([2, 7, 11, 15], 9)
```

    1
    0
    

복잡도도 줄었으며 실행시간 또한 48밀리초 밖에 되지 않는다. 

### Solution 4 조회 구조 개선

이번엔 하나의 for로 합쳐서 처리해보자.  
이 경우 전체를 모두 저장할 필요 없이 정답을 찾게 되면 함수를 바로 빠져나올 수 있다.  
그러나 두 번째 값을 찾기 위해 어차피 매번 비교해야 하기 때문에 앞서 풀이에 비해 성능상의 큰 이점은 없을 것 같다.


```python
from typing import List


class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        nums_map = {}
        # 하나의 `for`문으로 통합
        for i, num in enumerate(nums):
            if target - num in nums_map:
                return [nums_map[target - num], i]
            nums_map[num] = i

if __name__ == '__main__':
  s = Solution()
  print(s.twoSum([2, 7, 11, 15], 9))
```

    [0, 1]
    

실행속도는 44밀리초로 앞서 풀이와 큰 차이는 없으나, 코드는 한결 더 간결해졌다.

### Solution 5 투 포인터 이용

투 포인터란 왼쪽 포인터와 오른쪽 포인터의 합이 타겟보다 크다면 오른쪽 포인터를 왼쪽으로, 작다면 왼쪽 포인터를 오른쪽으로 옮기면서 값을 조정하는 방식이다.

![two_point](https://user-images.githubusercontent.com/72365663/102236253-a0978300-3f36-11eb-8a25-031a077413dc.JPG)


```python
from typing import List


class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        left, right = 0, len(nums) - 1    # left는 index 0에서 right는 '전체길이 - 1(문제에선 3)'에서 시작
        while not left == right:    # left와 right가 같지 않을 경우 계속 실행
            # 합이 타겟보다 크면 오른쪽 포인터를 왼쪽으로
            if nums[left] + nums[right] < target:
                left += 1
            # 합이 타겟보다 작으면 왼쪽 포인터를 오른쪽으로
            elif nums[left] + nums[right] > target:
                right -= 1
            else:
                return [left, right]

if __name__ == '__main__':
  s = Solution()
  print(s.twoSum([2, 7, 11, 15], 9))
```

    [0, 1]
    

코드도 간결하고 이해하기도 매우 쉽다.  
하지만 이코드의 경우 nums가 정렬된 상태가 아니라면 정렬하는 과정이 필요하다.  
그런데 이렇게 하면 인덱스가 모두 꼬여버리기 때문에 문제처럼 인덱스 값을 출력해야 하는 경우 사용할 수 없다.

## Q8 빗물트래핑

![rain](https://user-images.githubusercontent.com/72365663/102239482-3f71ae80-3f3a-11eb-98fd-5c32a8d7b1a3.JPG)
높이를 입력받아 비 온 후 얼마나 많은 물이 쌓일 수 있는지 계산하라.
- 입력
```
[0,1,0,2,1,0,1,3,2,1,2,1]
```
- 출력
```
6
```

### Solution 1 투 포인터를 최대로 이동
![rain_twopoint](https://user-images.githubusercontent.com/72365663/102240959-ca06dd80-3f3b-11eb-91d3-00c7c6c1f882.JPG)
위의 그림에서 가장 높이가 높은 막대는 높고 낮음에 무관하게, 전체 부피에 영향을 끼치지 않으면서 그저 왼쪽과 오른쪽을 가르는 장벽 역할을 한다.

![rain_twopoint2](https://user-images.githubusercontent.com/72365663/102293430-6d7edf00-3f8a-11eb-8063-2f1c50eca29b.JPG)


```python
from typing import List


class Solution:
    def trap(self, height: List[int]) -> int:
        if not height:
            return 0

        volume = 0
        left, right = 0, len(height) - 1    # left는 index 0에서 right는 '전체길이 - 1(문제에선 3)'에서 시작
        left_max, right_max = height[left], height[right]   # left와 right에 해당하는 value값

        while left < right:   # left와 right값이 같아지면 멈춤
            left_max, right_max = max(height[left], left_max), max(height[right], right_max)
            # 더 높은 쪽을 향해 투 포인터 이동    #과거와 현재 높이 중 너 높은 값을 출력
            if left_max <= right_max:
                volume += left_max - height[left]   # left값을 오른쪽으로 옮기면서 최대 높이에서 현재 높이를 빼면서 부피를 더해감
                left += 1
            else:
                volume += right_max - height[right]
                right -= 1
        return volume

if __name__ == '__main__':
  s = Solution()
  print(s.trap([0,1,0,2,1,0,1,3,2,1,2,1]))
```

    6
    

### Solutoin 2 스택 쌓기

![rain_stack](https://user-images.githubusercontent.com/72365663/102294994-cef47d00-3f8d-11eb-8fb4-125fe81c9cb7.JPG)

위의 그림과 같이 스택을 쌓아 나가면서 현재 높이가 이전 높이보다 높을 때, 즉 꺽이는 부분 변곡점을 기준으로 격차만큼 물 높이 volume를 채운다.  
이전 높이는 고정된 형태가 아니라 들쑥날쑥하기 때문에, 계속 스택으로 채워 나가다가 변곡점을 만날 때마다 스택에서 하나씩 꺼내면서 이전과의 차이만큼 물 높이를 채워 나간다.

- .append() : 선택된 요소의 마지막에 새로운 요소나 콘텐츠를 추가한다.
- .pop() : 배열의 맨 끝에 값을 제거한다.

![rain_stack2](https://user-images.githubusercontent.com/72365663/102301585-6a8bea80-3f9a-11eb-9ab0-1d31594ae1fd.png)


```python
from typing import List


class Solution:
    def trap(self, height: List[int]) -> int:
        stack = []
        volume = 0

        for i in range(len(height)):    # i는 0부터 리스트의 길이까지 범위, 즉 index를 말해준다
            # 변곡점을 만나는 경우
            while stack and height[i] > height[stack[-1]]:
                # 스택에서 꺼낸다
                top = stack.pop()
                print("top:",top)
                print("stack1:",stack)

                if not len(stack):    # stack 안이 비어있으면 멈춰라
                    break

                # 이전과의 차이만큼 물 높이 처리
                distance = i - stack[-1] - 1
                waters = min(height[i], height[stack[-1]]) - height[top]

                volume += distance * waters
                print("volume:", volume)

            stack.append(i)
            print("stack2:",stack)
        return volume

if __name__ == '__main__':
  s = Solution()
  print(s.trap([0,1,0,2,1,0,1,3,2,1,2,1]))
```

    stack2: [0]
    top: 0
    stack1: []
    stack2: [1]
    stack2: [1, 2]
    top: 2
    stack1: [1]
    volume: 1
    top: 1
    stack1: []
    stack2: [3]
    stack2: [3, 4]
    stack2: [3, 4, 5]
    top: 5
    stack1: [3, 4]
    volume: 2
    stack2: [3, 4, 6]
    top: 6
    stack1: [3, 4]
    volume: 2
    top: 4
    stack1: [3]
    volume: 5
    top: 3
    stack1: []
    stack2: [7]
    stack2: [7, 8]
    stack2: [7, 8, 9]
    top: 9
    stack1: [7, 8]
    volume: 6
    stack2: [7, 8, 10]
    stack2: [7, 8, 10, 11]
    6
    


```python
from typing import List


class Solution:
    def trap(self, height: List[int]) -> int:
        stack = []
        volume = 0

        for i in range(len(height)):    # i는 0부터 리스트의 길이-1까지 범위, 즉 index를 말해준다
            # 변곡점을 만나는 경우
            while stack and height[i] > height[stack[-1]]:
                # 스택에서 꺼낸다
                top = stack.pop()

                if not len(stack):
                    break

                # 이전과의 차이만큼 물 높이 처리
                distance = i - stack[-1] - 1
                waters = min(height[i], height[stack[-1]]) - height[top]

                volume += distance * waters

            stack.append(i)
        return volume

if __name__ == '__main__':
  s = Solution()
  print(s.trap([0,1,0,2,1,0,1,3,2,1,2,1]))
```

    6
    

## Q9 세 수의 합

배열을 입력받아 합으로 0을 만들 수 있는 3개의 엘리먼트를 출력하라.
- 입력
```
nums = [-1, 0, 1, 2, -1, -4]
```
- 출력
```
[
    [-1, 0, 1],
    [-1, -1, 2]
]
```

### Solution 1 브루트 포스로 계산

앞뒷로 같은 값이 있을 경우, 이를 쉽게 처리하기 위해 먼저 다음과 같이 sort()함수를 사용해 정렬부터 한다.  
`nums.sort()`  

[-1, 0, 1, 2, -1, -4]을 정렬한 결과 [-4, -1, -1, 0, 1, 2]  

![3num_1](https://user-images.githubusercontent.com/72365663/102305719-0ff78c00-3fa4-11eb-97bf-eafd227d2a4f.JPG)  

이 그림에서 i, j, k 각각의 포인터가 계속 이동하면서 i+j+k=0을 찾아낸다.  
브루트 포스 풀이에는 중복된 값이 있을 수 있으므로 이경우 다음과 같이 continue로 건너뛰도록 처리한다.

```
if i > 0 and nums[i] == nums[i - 1]:
    continue
```


```python
from typing import List


class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        results = []
        nums.sort()   # 리스트 nums를 정렬

        # 브루트 포스 n^3 반복
        for i in range(len(nums) - 2):
            # 중복된 값 건너뛰기
            if i > 0 and nums[i] == nums[i - 1]:    # i가 0보다 크고, 앞의 값과 같을 경우
                continue    # 아래 코드를 실행하지 않고 건너뜀
            for j in range(i + 1, len(nums) - 1):
                if j > i + 1 and nums[j] == nums[j - 1]:
                    continue
                for k in range(j + 1, len(nums)):
                    if k > j + 1 and nums[k] == nums[k - 1]:
                        continue
                    if nums[i] + nums[j] + nums[k] == 0:
                        results.append([nums[i], nums[j], nums[k]])

        return results

if __name__ == '__main__':
  s = Solution()
  print(s.threeSum([-1, 0, 1, 2, -1, -4]))
```

    [[-1, -1, 2], [-1, 0, 1]]
    

### Solution 2 투 포인터로 합 계산

i를 축으로 하고, 중복된 값을 건너뛰게 한 부분은 다음과 같이 앞서 풀이와 동일하다.  
앞선 풀이와의 차이점은 i의 다음 지점과 마지막 지점을 이 그림과 같이 left, right로 설정하고 간격을 좁혀가며 sum을 계산하는 부분이다.  
투 포인터가 간격을 좁혀나가며 합 sum을 계산한다.

![3num_2](https://user-images.githubusercontent.com/72365663/102307668-4931fb00-3fa8-11eb-9fd3-63b7f4cb71cc.JPG)


```python
from typing import List


class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        results = []
        nums.sort()

        for i in range(len(nums) - 2):
            # 중복된 값 건너뛰기
            if i > 0 and nums[i] == nums[i - 1]:
                continue

            # 간격을 좁혀가며 합 `sum` 계산
            left, right = i + 1, len(nums) - 1    # left는 i의 다음 지점, right를 마지막 지점으로 설정
            while left < right:   # left값이 right 값보다 작지 않을 때까지 계속 수행한다
                sum = nums[i] + nums[left] + nums[right]
                if sum < 0:   # 합이 0보다 작을 경우 left를 오른쪽으로 한 칸 이동
                    left += 1
                elif sum > 0:   # 합이 0보다 클 경우 right를 왼쪽으로 한 칸 이동
                    right -= 1
                else:
                    # `sum = 0`인 경우이므로 정답 및 스킵 처리  # 동일 값이 존재하여 중복값이 나올 수 있으므로 스킵을 해준다
                    results.append([nums[i], nums[left], nums[right]])

                    while left < right and nums[left] == nums[left + 1]:    # left값을 계속 증가, right값을 계속 감소시켜 left>right로 만들어 종료
                        left += 1
                    while left < right and nums[right] == nums[right - 1]:
                        right -= 1
                    left += 1
                    right -= 1

        return results

if __name__ == '__main__':
  s = Solution()
  print(s.threeSum([-1, 0, 1, 2, -1, -4]))
```

    [[-1, -1, 2], [-1, 0, 1]]
    

## Q10 배열 파티션 1

n개의 페어를 이용한 min(a, b)의 합으로 만들 수 있는 가장 큰 수를 출력하라.

- 입력
```
[1,4,3,2]
```
- 출력
```
4
```
- 설명
    - n은 2가 되며, 최대 합은 4이다.
    - min(1,2) + min(3,4) = 4


### Solution 1 오름차순 풀이

페어의 min()을 합산했을 때 최대를 만드는 것은 결국 min()이 되도록 커야 한다는 뜻이고, 뒤에서부터 내림차순으로 집어넣으면 항상 최대 min() 페어를 유지할 수 있다.  
이 문제에서 배열 입력값은 항상 2n개일 것이므로 앞에서부터 오름차순으로 집어넣어도 결과는 같을 것이다.  
![Partition_1](https://user-images.githubusercontent.com/72365663/102313753-8f8d5700-3fb4-11eb-80af-232610d0eb05.JPG)  

그림에서 처럼 하나씩 조합해보면 다음과 같다.  
  1.min(1,4) + min(2,3) = 3  
  2.min(1,3) + min(2,4) = 3  
  3.min(1,2) + min(3,4) = 4  

정렬된 상태에서 앞에서부터 오름차순으로 인접 요소 페어를 만들면 가장 큰 a1과 a2 페어들을 만들 수 있고 이 페어들의 합이 곧 만들 수 있는 최대 합이 된다.  


```python
from typing import List


class Solution:
    def arrayPairSum(self, nums: List[int]) -> int:
        sum = 0
        pair = []
        nums.sort()   # 오름차순으로 정렬

        for n in nums:
            # 앞에서 부터 오름차순으로 페어를 만들어 합 계산
            pair.append(n)    # n값을 작은 것부터 페어에 추가
            if len(pair) == 2:    # pair 리스트의 길이가 2가 되면 sum에 min(pair)값을 더한다
                sum += min(pair)
                print("pair:",pair)
                pair = []

        return sum

if __name__ == '__main__':
  s = Solution()
  print(s.arrayPairSum([1,4,3,2]))
```

    pair: [1, 2]
    pair: [3, 4]
    4
    

### Solution 2 짝수 번째 값 계산

이렇게 페어에 대해 일일이 min()값을 구하지 않아도 짝수 번째 값(index는 0부터 시작하므로)을 더하면 될 것 같다.  
정렬된 상태에서는 짝수 번째에 항상 작은 값이 위치하기 때문이다.
이렇게 하면 코드를 매우 간결하게 구현할 수 있다.


```python
from typing import List


class Solution:
    def arrayPairSum(self, nums: List[int]) -> int:
        sum = 0
        nums.sort()

        for i, n in enumerate(nums):
            # 짝수 번째 값의 합 계산
            if i % 2 == 0:
                sum += n

        return sum

if __name__ == '__main__':
  s = Solution()
  print(s.arrayPairSum([1,4,3,2]))
```

    4
    

### Solution 3 파이썬다운 방식

이 코드를 한 줄로 풀이해보자.  
슬라이싱 구문 `::2`는 2칸씩 건너뛰므로 짝수 번째를 계산하는 것과 동일하다.


```python
from typing import List


class Solution:
    def arrayPairSum(self, nums: List[int]) -> int:
        return sum(sorted(nums)[::2])

if __name__ == '__main__':
  s = Solution()
  print(s.arrayPairSum([1,4,3,2]))
```

    4
    

## Q11 자신을 제외한 배열의 곱

배열을 입력받아 output[i]가 자신을 제외한 나머지 모든 요소의 곱셈 결과가 되도록 출력하라.  
- 입력
```
[1,2,3,4]
```
- 출력
```
[24,12,8,6]
```
- 주의
    - 나눗셈을 하지 않고 O(n)에 풀이하라.
    - 즉, 미리 전체 곱셈 값을 구해놓고 각 항목별로 자기자신을 나눠서 풀이하는 방법은 안된다는 뜻이다.

### Solution 1 왼쪽 곱셈 결과에 오른쪽 값을 차례대로 곱셈

자기 자신을 제외하고 왼쪽의 곱셈 결과와 오른쪽의 곱셈 결과를 곱해야 한다.  
![Product_of_Array](https://user-images.githubusercontent.com/72365663/102321999-c158ea80-3fc1-11eb-8de4-0e2785b794b6.JPG)

![Product_of_Array2](https://user-images.githubusercontent.com/72365663/102351815-67b8e600-3fea-11eb-9cd2-5f006b0e8708.JPG)


```python
from typing import List


class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        out = []
        p = 1
        # 왼쪽 곱셈
        for i in range(0, len(nums)):   # 먼저 왼쪽부터 곱해서 result에 추가한다.
            out.append(p)
            p = p * nums[i]
            print("out1:",out)
        p = 1
        # 왼쪽 곱셈 결과에 오른쪽 값을 차례대로 곱셈
        for i in range(len(nums) - 1, 0 - 1, -1):   # 오른쪽부터(역순) out값에 곱해준다
            out[i] = out[i] * p
            p = p * nums[i]
            print("out2:",out)
        return out

if __name__ == '__main__':
  s = Solution()
  print(s.productExceptSelf([1,2,3,4]))
```

    out1: [1]
    out1: [1, 1]
    out1: [1, 1, 2]
    out1: [1, 1, 2, 6]
    out2: [1, 1, 2, 6]
    out2: [1, 1, 8, 6]
    out2: [1, 12, 8, 6]
    out2: [24, 12, 8, 6]
    [24, 12, 8, 6]
    

## Q12 주식을 사고팔기 가장 좋은 시점

한 번의 거래로 낼 수 있는 최대 이익을 산출하라.
- 입력
```
[7,1,5,3,6,4]
```
- 출력
```
5
```
- 설명
    - 저점에 사서 고점에 팔아서, 낼 수 있는 최대 이익을 찾는 문제이다.
    - 1일 때 사서 6일 때 팔면 5의 이익을 얻는다.

### Solution 1 브루트 포스로 계산


```python
from typing import List


class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        max_price = 0

        for i, price in enumerate(prices):    # i는 인덱스 값, price는 value값으로 지정
            for j in range(i, len(prices)):   
                max_price = max(prices[j] - price, max_price)   #일일이 다 빼서 최고값만 남겨놓는다.

        return max_price

if __name__ == '__main__':
  s = Solution()
  print(s.maxProfit([7,1,5,3,6,4]))
```

    5
    

### Solution 2 저점과 현재 값과의 차이 계산

![stock_1](https://user-images.githubusercontent.com/72365663/102354730-4fe36100-3fee-11eb-983b-9f0febe21929.JPG)

인덱스1은 저점을 가리키고 있고 인덱스4는 고점을 가리킨다.  
여기서는 현재값을 가리키는 포인터가 우측으로 이동하면서 이전 상태의 저점을 기준으로 가격 차이를 계산하고, 만약 클 경우 최댓값을 계속 교체해나가는 형태로 풀고 있다.


```python
import sys
from typing import List


class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        profit = 0
        min_price = sys.maxsize   # int값에서 가능한 최댓값을 불러온다다
        print("maxsize:",min_price)

        # 최소값과 최대값 계속 갱신
        for price in prices:    # prices에서 value값을 차례대로 불러온다
            min_price = min(min_price, price)
            profit = max(profit, price - min_price)
            print("min_price:",min_price)
            print("profit:",profit)

        return profit

if __name__ == '__main__':
  s = Solution()
  print(s.maxProfit([7,1,5,3,6,4]))
```

    maxsize: 9223372036854775807
    min_price: 7
    profit: 0
    min_price: 1
    profit: 0
    min_price: 1
    profit: 4
    min_price: 1
    profit: 4
    min_price: 1
    profit: 5
    min_price: 1
    profit: 5
    5
    
