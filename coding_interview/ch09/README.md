# Chap 9 스택, 큐

스택(Stack)과 큐(Queue)는 프로그래밍이라는 개념이 탄생할 때부터 사용된 가장 고전적인 자료구조로, 자료를 담아두는 자료형이다.  
**스택은 LIFO(Last-In-First-Out, 후입선출)**, **큐는 FIFO(First-In-First-Out, 선입선출)**로 처리된다.  
파이썬은 스택 자료형을 별도로 제공하지는 않지만, 리스트가 사실상 스택의 모든 연산을 지원한다.  
큐 또한 리스트가 큐의 모든 연산을 지원한다. 다만 리스트는 동적 배열로 구현되어 있어 큐의 연산을 수행하기에는 효율적이지 않기 때문에, 큐를 위해서는 데크(Deque)라는 별도의 자료형을 사용해야 좋은 성능을 낼 수 있다.  
리스트는 스택과 큐의 모든 연산을 지원하기 때문에 사실상 리스트를 잘 사용하기만 해도 충분하다.
![stack](https://user-images.githubusercontent.com/72365663/102454044-b31dd380-4080-11eb-8522-47c8cd4f4927.png)
![queue](https://user-images.githubusercontent.com/72365663/102454049-b44f0080-4080-11eb-932e-f02011b7527f.png)



## 스택
스택은 다음과 같은 2가지 주요 연산을 지원하는 요소의 컬렉션으로 사용되는 추상 자료형이다.
- `push()`: 요소를 컬렉션에 추가한다.
- `pop()`: 아직 제거되지 않은 가장 최근에 삽입된 요소를 제거한다.



### 연결 리스트를 이용한 스택 ADT 구현

연결 리스트를 이ㅛㅇ해 실제로 스택을 한번 구현해보자.  
먼저 다음과 같이 연결 리스트를 담을 Node 클래스부터 정의한다.


```python
class Node:
  def __init__(self, item, next):
    self.item = item
    self.next = next
```

초기화 함수 `__init__()`에서 노드의 값은 `item`으로, 다음 노드를 가리키는 포인터는 `next`로 정의한다.  
이제 스택의 연산인 `push()`와 `pop()`을 담은 Stack 클래스를 다음과 같이 정의한다.


```python
class Stack:
  def __init__(self):
      self.last = None

  def push(self, item):
     self.last = Node(item, self.last)
  
  def pop(self):
        item = self.last.item
        self.last = self.last.next
        return item
```

`push()`는 연결 리스트에 요소를 추가하면서 가장 마지막 값을 next로 지정하고, 포인터인 last는 가장 마지막으로 이동시킨다.  
`pop()`은 가장 마지막 아이템을 끄집어내고 last 포인터를 한 칸 앞으로 전진시킨다. 즉 이전에 추가된 값을 가리키게 한다.  
이제 다음과 같이 1부터 5까지의 값을 스택에 입력해보자.


```python
stack = Stack()
stack.push(1)
stack.push(2)
stack.push(3)
stack.push(4)
stack.push(5)
```

스택 변수 stack의 입력된 값을 도식화하면 아래 그림과 같다.  
![stack](https://user-images.githubusercontent.com/72365663/102710304-c76a0680-42f4-11eb-97a3-13c7473c0bc3.JPG)  
stack은 각각 이전 값을 가리키는 연결 리스트로 구현되어 있으며, 가장 마지막 값은 last 포인터가 가리킨다.  
이제 `pop()` 메소드로 스택의 값을 차례대로 출력해보자.


```python
  for _ in range(5):
    print(stack.pop())
```

    5
    4
    3
    2
    1
    

가장 최근에 입력된 순서대로(LIFO) 출력되는 것을 확인할 수 있다.

## Q20 유효한 괄호

괄호로 된 입력값이 올바른지 판별하라

- 입력
```
(){}[]
```

- 출력
```
True
```

### Solution 1 스택 일치 여부 판별



```python
# Dictionary 조회 원리
a = {'Korea': 'Seoul', 
     'Canada': 'Ottawa', 
     'USA': 'Washington D.C'}

print("Korea in a: ","Korea" in a)    # Key는 바로 조회 가능
print("Seoul in a: ","Seoul" in a)    # Value는 바로 조회 불가
print("Value값 조회: ", a["Korea"])   # Key값을 입력하면 Value값 추출
```

    Korea in a:  True
    Seoul in a:  False
    Value값 조회:  Seoul
    

![ValidParen](https://user-images.githubusercontent.com/72365663/102711632-aad2cc00-42fe-11eb-99e3-fad2e085cb19.JPG)


```python
class Solution:
    def isValid(self, s: str) -> bool:
        stack = []
        # 괄호로 이루어진 dictionary 생성
        table = {
            ')': '(',
            '}': '{',
            ']': '[',
        }
        
        # 스택 이용 예외 처리 및 일치 여부 판별
        for char in s:
            if char not in table:
                stack.append(char)    # stack에 table 키에 없는 char 추가 
            elif not stack or table[char] != stack.pop():   # stack의 마지막 값을 추출해내서 비교
                return False
        return len(stack) == 0
        

if __name__ == '__main__':
  s = Solution()
  print(s.isValid("()[]{}"))
  print(s.isValid(()))
  print(s.isValid({}))
  print(s.isValid([]))
```

    True
    True
    True
    True
    

## Q21 중복 문자 제거

중복된 문자를 제외하고 사전식 순서로 나열하라  
  
예제1
- 입력
```
"bcabc"
```
- 출력
```
"abc"
```
  
예제2
- 입력
```
"cbacdcbc"
```
- 출력
```
"acdb"
```

- 설명
    - 사전식 순서란 글자 그대로 사전에서 가장 먼저 찾을 수 있는 순서를 말한다.
    - bcabc에서 중복 문자를 제거하면 앞에 bc가 제거되고 사전식 순서대로 abc가 남을 것이다.
    - ebcabc의 경우 순서상으로는 abce가 맞지만, e는 딱 한 번만 등장하여 중복문자가 아니므로 제거할 수가 없기 때문에 eabc가 출력되게 된다.
    - ebcabce라면 첫 번째 e는 중복으로 제거할 수 있기 때문에 abce가 출력된다.  
![removeletter](https://user-images.githubusercontent.com/72365663/102712126-ecb14180-4301-11eb-8cd1-43228e06da06.JPG)


### Solution 1 재귀를 이용한 분리

![removeletter2](https://user-images.githubusercontent.com/72365663/102713357-0c009c80-430b-11eb-8686-1e249b9938ac.JPG)


```python
s = "cbacdcbc"
print(set(s))   # 중복 문자 제거
print(sorted(set(s)))   # 문자 정렬
print(s[s.index("a"):])   # 해당 문자부터 끝까지
```

    {'d', 'c', 'b', 'a'}
    ['a', 'b', 'c', 'd']
    acdcbc
    


```python
class Solution:
    def removeDuplicateLetters(self, s: str) -> str:
        # 집합으로 정렬
        for char in sorted(set(s)):   # 중복 문자를 제외한 알파벳 순으로 문자열 입력값을 모두 정렬
            suffix = s[s.index(char):]
            print("suffix:", suffix)

            # 전체 집합과 접미사 집합이 일치할때 분리 진행
            if set(s) == set(suffix):
                return char + self.removeDuplicateLetters(suffix.replace(char, ''))   # 해당문자 + suffix에서 해당문자 제거한 문자
            print("s:",s)
        return ''

if __name__ == '__main__':
  s = Solution()
  print(s.removeDuplicateLetters("bcabc"))
  print(s.removeDuplicateLetters("cbacdcbc"))
```

    suffix: abc
    suffix: bc
    suffix: c
    abc
    suffix: acdcbc
    suffix: bc
    s: cdcbc
    suffix: cdcbc
    suffix: b
    s: db
    suffix: db
    suffix: b
    acdb
    

### Solution 2 스택을 이용한 문자 제거


```python
import collections

s = "cbacdcbc"

print(collections.Counter(s))   # 중복문자가 몇개인지 알려줌
```

    Counter({'c': 4, 'b': 2, 'a': 1, 'd': 1})
    


```python
print('a' < 'b')
print('d' < 'c')
```

    True
    False
    

![removeletter3](https://user-images.githubusercontent.com/72365663/102720827-01113080-433a-11eb-909e-3d29bf2c549d.png)


```python
import collections

class Solution:
    def removeDuplicateLetters(self, s: str) -> str:
        counter, seen, stack = collections.Counter(s), set(), []

        for char in s:
            counter[char] -= 1
            if char in seen:    # char이 seen에 속해 있을 경우 스킵
                continue

            # 뒤에 붙일 문자가 남아 있다면 스택에서 제거
            while stack and char < stack[-1] and counter[stack[-1]] > 0:
                seen.remove(stack.pop())
            stack.append(char)
            seen.add(char)
            print("stack:",stack)
            print("seen2:",seen)

        return ''.join(stack)

if __name__ == '__main__':
  s = Solution()
  print(s.removeDuplicateLetters("bcabc"))
  print(s.removeDuplicateLetters("cbacdcbc"))
```

    stack: ['b']
    seen2: {'b'}
    stack: ['b', 'c']
    seen2: {'c', 'b'}
    stack: ['a']
    seen2: {'a'}
    stack: ['a', 'b']
    seen2: {'b', 'a'}
    stack: ['a', 'b', 'c']
    seen2: {'c', 'b', 'a'}
    abc
    stack: ['c']
    seen2: {'c'}
    stack: ['b']
    seen2: {'b'}
    stack: ['a']
    seen2: {'a'}
    stack: ['a', 'c']
    seen2: {'c', 'a'}
    stack: ['a', 'c', 'd']
    seen2: {'c', 'd', 'a'}
    stack: ['a', 'c', 'd', 'b']
    seen2: {'b', 'c', 'd', 'a'}
    acdb
    

## Q22 일일 온도

매일의 화씨 온도(F) 리스트 T를 입력받아서, 더 따뜻한 날씨를 위해서는 며칠을 더 기다려야 하는지를 출력하라.

- 입력
```
T = [73, 74, 75, 71, 69, 72, 76, 73]
```

- 출력
```
[1, 1, 4, 2, 1, 1, 0, 0]
```

- 설명
첫째 날(73도)에서 더 따뜻한 날을 위해서는 하루만 기다리면 된다. 셋째 날(75도)에서 더 따뜻한 날을 기다리기 위해서는(71, 69, 72, 76) 4일을 기다려야 한다.

### Solution 1 스택 값 비교

![dailytemp](https://user-images.githubusercontent.com/72365663/102715607-f181ef80-4319-11eb-97b1-cc6509335479.JPG)  
현재의 인덱스를 계속 스택에 쌓아두다가, 이전보다 상승하는 지점에서 현재 온도와 스택에 쌓아둔 인덱스 지점의 온도 차이를 비교해서, 더 높다면 스택의 값을 pop으로 꺼내고 현재 인덱스와 스택에 쌓아둔 인덱스의 차이를 정답으로 처리한다.  
![dailytemp2](https://user-images.githubusercontent.com/72365663/102716286-432c7900-431e-11eb-9798-b296dd7ea5ed.png)  


```python
from typing import List


class Solution:
    def dailyTemperatures(self, T: List[int]) -> List[int]:
        answer = [0] * len(T)
        stack = []
        print("answer_initial:", answer)
        for i, cur in enumerate(T):
            # 현재 온도가 스택 값보다 높다면 정답 처리
            while stack and cur > T[stack[-1]]:
                last = stack.pop()
                answer[last] = i - last
            stack.append(i)
            print("answer:", answer)

        return answer

if __name__ == '__main__':
  s = Solution()
  print(s.dailyTemperatures([73, 74, 75, 71, 69, 72, 76, 73]))
```

    answer_initial: [0, 0, 0, 0, 0, 0, 0, 0]
    answer: [0, 0, 0, 0, 0, 0, 0, 0]
    answer: [1, 0, 0, 0, 0, 0, 0, 0]
    answer: [1, 1, 0, 0, 0, 0, 0, 0]
    answer: [1, 1, 0, 0, 0, 0, 0, 0]
    answer: [1, 1, 0, 0, 0, 0, 0, 0]
    answer: [1, 1, 0, 2, 1, 0, 0, 0]
    answer: [1, 1, 4, 2, 1, 1, 0, 0]
    answer: [1, 1, 4, 2, 1, 1, 0, 0]
    [1, 1, 4, 2, 1, 1, 0, 0]
    

## 큐

큐(Queue)는 시퀀스의 한쪽 끝에는 엔티티를 추가하고, 다른 반대쪽 끝에는 제거할 수 있는 엔티티 컬렉션이다.  

## Q23 큐를 이용한 스택 구현

큐를 이용해 다음 연산을 지원하는 스택을 구현하라.

- push(x): 요소x를 스택에 삽입한다.
- pop(): 스택의 첫 번째 요소를 삭제한다.
- top():스택의 첫 번째 요소를 가져온다.
- empty(): 스택이 비어 있는지 여부를 리턴한다.
  
  
MyStack stack = new MyStack();

stack.push(1);  
stack.push(2);  
stack.top();    // 2 리턴  
stack.pop();    // 2 리턴  
stack.empty()    // false 리턴  

### Solution 1 `push()`할 때 큐를 이용해 재정렬

- popleft(): 가장 왼쪽에 있는 값을 뺀다.


```python
import collections

class MyStack:
    def __init__(self):
        self.q = collections.deque()

    def push(self, x):
        self.q.append(x)
        # 요소 삽입 후 맨 앞에 두는 상태로 재정렬
        for _ in range(len(self.q) - 1):
            self.q.append(self.q.popleft())   # 가장 왼쪽에 있는 값을 오른쪽에 넣어준다. [1,2] -> [2,1]
        print("큐:", self.q)

    def pop(self):
        return self.q.popleft()

    def top(self):
        return self.q[0]

    def empty(self):
        return len(self.q) == 0

if __name__ == '__main__':
  stack = MyStack()
  print(stack.push(1))
  print(stack.push(2))
  print(stack.top())
  print(stack.pop())
  print(stack.empty())
```

    큐: deque([1])
    None
    큐: deque([2, 1])
    None
    2
    2
    False
    

## Q24 스택을 이용한 큐 구현

스택을 이용해 다음 연산을 지원하는 큐를 구현하라.

- push(x): 요소x를 큐 마지막에 삽입한다.
- pop(): 큐 처음에 있는 요소를 제거한다.
- peek():큐 처음에 있는 요소를 조회한다.
- empty(): 큐이 비어 있는지 여부를 리턴한다.
  
```
MyQueue queue = new MyQueue();  
  
queue.push(1);  
queue.push(2);  
queue.peek();  // 1 리턴  
queue.pop();   // 1 리턴  
queue.empty()  // false 리턴  
```

### Solution 1 스택 2개 사용

![que](https://user-images.githubusercontent.com/72365663/102717338-fef0a700-4324-11eb-94a8-8be073487ecd.JPG)


```python
class MyQueue:
    def __init__(self):
        self.input = []
        self.output = []

    def push(self, x):
        self.input.append(x)

    def pop(self):
        self.peek()
        return self.output.pop()

    def peek(self):
        # output이 없으면 모두 재입력
        if not self.output:
            while self.input:
                self.output.append(self.input.pop())    # input 스택에 있는 오른쪽 값부터 output 스택에 넣는다
        return self.output[-1]

    def empty(self):
        return self.input == [] and self.output == []

if __name__ == '__main__':
  queue = MyQueue()
  print(queue.push(1))
  print(queue.push(2))
  print(queue.peek())
  print(queue.pop())
  print(queue.empty())
```

    None
    None
    1
    1
    False
    

## Q25 원형 큐 디자인

원형 큐를 디자인하라.

```
MyCircularQueue circularQueue = new MyCircularQueue(5);  //크기를 5로 지정  
circularQueue.enQueue(10);  //  true 리턴  
circularQueue.enQueue(20);  //  true 리턴  
circularQueue.enQueue(30);  //  true 리턴  
circularQueue.enQueue(40);  //  true 리턴  
circularQueue.Rear();       //  40 리턴  
circularQueue.isFull();     //  false 리턴  
circularQueue.deQueue();    //  true 리턴  
circularQueue.deQueue();    //  true 리턴  
circularQueue.enQueue(50);  //  true 리턴  
circularQueue.enQueue(60);  //  true 리턴  
circularQueue.Rear();       //  60 리턴  
circularQueue.Front();      //  30 리턴  
```
  
'원형 큐'는 기존의 큐처럼 FIFO 구조를 지니면서, 마지막 위치가 시작 위치와 연결되는 아래의 그림과 같은 원형구조를 띠는 큐이다.  
기존의 큐는 공간이 꽉 차게 되면 더 이상 요소를 추가할 수 없었다. 앞쪽에 요소들이 deQueue()로 모두 빠져서 충분한 공간이 남게 돼도 그쪽으로는 추가할 수 있는 방법이 없다.  
그래서 앞쪽에 공간이 남아 있다면 그림처럼 동그랗게 연결해 앞쪽으로 추가할 수 있도록 재활용 가능한 구조가 바로 원형 큐이다.
![circularQ](https://user-images.githubusercontent.com/72365663/102717955-c5ba3600-4328-11eb-97dd-373476c65649.png)

### Solution 1 배열을 이용한 풀이


```python
class MyCircularQueue:
    def __init__(self, k: int):
        self.q = [None] * k
        self.maxlen = k   # 최대길이
        self.p1 = 0   # front
        self.p2 = 0   # rear
        print("q:",self.q)

    # enQueue(): 리어(rear) 포인터 이동
    def enQueue(self, value: int) -> bool:
        if self.q[self.p2] is None:
            self.q[self.p2] = value   # rear위치의 값이 비어있을 경우 입력된 value 값으로 채운다
            self.p2 = (self.p2 + 1) % self.maxlen   # rear의 위치를 다음칸으로 이동시킨다(전체길의 나머지 이므로 5의 경우 0, 6의 경우 1 출력)
            return True
        else:
            return False

    # deQueue(): 프론트(front) 포인터 이동
    def deQueue(self) -> bool:
        if self.q[self.p1] is None:
            return False
        else:
            self.q[self.p1] = None    # 기존 front 위치의 값을 None으로 바꾼다(지워버린다)
            self.p1 = (self.p1 + 1) % self.maxlen   # front의 위치를 다음칸으로 이동시킨다
            return True

    def Front(self) -> int:
        return -1 if self.q[self.p1] is None else self.q[self.p1]   # 현재 front 위치의 값 출력

    def Rear(self) -> int:
        return -1 if self.q[self.p2 - 1] is None else self.q[self.p2 - 1]   # rear은 현재 값이 채워져 있는 다음의 빈 칸에 위치해 있으므로 현위치에서 앞으로 이동해야 한다.

    def isEmpty(self) -> bool:
        return self.p1 == self.p2 and self.q[self.p1] is None   # rear와 front가 같은 위치에 있으며 해당값이 None일 경우 비어있는 걸로 간주

    def isFull(self) -> bool:
        return self.p1 == self.p2 and self.q[self.p1] is not None   # rear와 front가 같은 위치에 있으며 해당값이 None이 아닐 경우 꽉 차있는 걸로 간주


if __name__ == '__main__':
  circularQueue = MyCircularQueue(5)    # 크기를 5로 지정
  print(circularQueue.enQueue(10))
  print(circularQueue.enQueue(20))
  print(circularQueue.enQueue(30))
  print(circularQueue.enQueue(40))
  print(circularQueue.Rear())
  print(circularQueue.isFull())
  print(circularQueue.deQueue()) 
  print(circularQueue.deQueue()) 
  print(circularQueue.enQueue(50)) 
  print(circularQueue.enQueue(60))
  print(circularQueue.Rear())
  print(circularQueue.Front())
```

    q: [None, None, None, None, None]
    True
    True
    True
    True
    40
    False
    True
    True
    True
    True
    60
    30
    
