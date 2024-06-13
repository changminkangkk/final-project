# data_collection.py
import os
import time
from github import Github, GithubException
from github.Repository import Repository

def save_chatgpt_code():
    chatgpt_code_examples = [
        """
        def add(a, b):
            return a + b
        """,
        """
        class Dog:
            def __init__(self, name, age):
                self.name = name
                self.age = age

            def bark(self):
                print("Woof!")
        """,
        """
        import numpy as np
        def create_matrix(n):
            return np.zeros((n, n))
        """,
        """
        def greet(name):
            return f"Hello, {name}!"
        """,
        """
        def calculate_area_of_circle(radius):
            return 3.14 * radius ** 2
        """,
        """
        def greet(name):
            return f"Hello, {name}!"
        """,
        """
        def is_prime(number):
            if number <= 1:
                return False
            for i in range(2, int(number ** 0.5) + 1):
                if number % i == 0:
                 return False
            return True
        """,
        """
        def reverse_string(s):
            return s[::-1]
        """,
        """
        def factorial(n):
            if n == 0 or n == 1:
                return 1
            else:
                return n * factorial(n - 1)
        """,
        """
        def square_root(x):
            return x ** 0.5
        """,
        """
        def is_even(number):
            return number % 2 == 0
        """,
        """
        def count_occurrences(lst, element):
            return lst.count(element)
        """,
        """
        def convert_to_celsius(fahrenheit):
            return (fahrenheit - 32) * 5 / 9
        """,
        """
        def generate_fibonacci_sequence(n):
            sequence = [0, 1]
            while len(sequence) < n:
                sequence.append(sequence[-1] + sequence[-2])
            return sequence
        """,
        """
        def remove_duplicates(lst):
            return list(set(lst))
        """,
        """
        def is_valid_email(email):
            return "@" in email and "." in email
        """,
        """
        def find_largest_number(lst):
            return max(lst)
        """,
        """
        def calculate_factorial(n):
            result = 1
            for i in range(1, n + 1):
                result *= i
            return result
        """,
        """
        def find_median(lst):
            sorted_lst = sorted(lst)
            n = len(sorted_lst)
            if n % 2 == 0:
                return (sorted_lst[n // 2 - 1] + sorted_lst[n // 2]) / 2
            else:
                return sorted_lst[n // 2]
        """,
        """
        def calculate_hypotenuse(a, b):
            return (a ** 2 + b ** 2) ** 0.5
        """,
        """
        def reverse_list(lst):
            return lst[::-1]
        """,
        """
        def is_leap_year(year):
            return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
        """,
        """
        def find_common_elements(lst1, lst2):
            return list(set(lst1) & set(lst2))
        """,
        """
        def remove_whitespace(text):
            return text.replace(" ", "")
        """,
        """
        def generate_prime_numbers(n):
            primes = []
            for num in range(2, n + 1):
                if all(num % i != 0 for i in range(2, int(num ** 0.5) + 1)):
                    primes.append(num)
            return primes
        """,
        """
        def calculate_average(lst):
            return sum(lst) / len(lst)
        """,
        """
        def remove_punctuation(text):
            return "".join(char for char in text if char.isalnum())
        """,
        """
        def find_gcd(a, b):
            while b:
                a, b = b, a % b
            return a
        """,
        """
        def is_palindrome(word):
            return word == word[::-1]
        """,
        """
        def remove_duplicates(lst):
            return list(dict.fromkeys(lst))
        """,
        """
        def calculate_perimeter_of_rectangle(length, width):
            return 2 * (length + width)
        """,
        """
        def findMedianSortedArrays(nums1, nums2):
            m, n = len(nums1), len(nums2)
    
            # nums1의 길이가 항상 더 길도록 스왑
            if m > n:
                nums1, nums2, m, n = nums2, nums1, n, m
    
            imin, imax, half_len = 0, m, (m + n + 1) // 2
    
            while imin <= imax:
                i = (imin + imax) // 2
                j = half_len - i
                
                if i < m and nums2[j-1] > nums1[i]:
                    # i가 작은 경우, i를 증가시켜서 j를 줄입니다.
                    imin = i + 1
                elif i > 0 and nums1[i-1] > nums2[j]:
                    # i가 큰 경우, i를 감소시켜서 j를 늘립니다.
                    imax = i - 1
                else:
                    # 적절한 i를 찾았습니다.
                    if i == 0: max_of_left = nums2[j-1]
                    elif j == 0: max_of_left = nums1[i-1]
                    else: max_of_left = max(nums1[i-1], nums2[j-1])
                    
                    if (m + n) % 2 == 1:
                        return max_of_left
                    
                    if i == m: min_of_right = nums2[j]
                    elif j == n: min_of_right = nums1[i]
                    else: min_of_right = min(nums1[i], nums2[j])
                    
                    return (max_of_left + min_of_right) / 2.0
        
        # Test cases
        nums1_1, nums2_1 = [1, 3], [2]
        nums1_2, nums2_2 = [1, 2], [3, 4]
        
        print(findMedianSortedArrays(nums1_1, nums2_1))  # Output: 2.00000
        print(findMedianSortedArrays(nums1_2, nums2_2))  # Output: 2.50000
        """,
        """
        class ListNode:
            def __init__(self, val=0, next=None):
                self.val = val
                self.next = next
        
        def mergeKLists(lists):
            if not lists:
                return None
            
            def mergeTwoLists(l1, l2):
                dummy = ListNode()
                curr = dummy
                
                while l1 and l2:
                    if l1.val < l2.val:
                        curr.next = l1
                        l1 = l1.next
                    else:
                        curr.next = l2
                        l2 = l2.next
                    curr = curr.next
                
                curr.next = l1 if l1 else l2
                return dummy.next
            
            while len(lists) > 1:
                merged_lists = []
                for i in range(0, len(lists), 2):
                    l1 = lists[i]
                    l2 = lists[i+1] if i+1 < len(lists) else None
                    merged_lists.append(mergeTwoLists(l1, l2))
                lists = merged_lists
            
            return lists[0]
        
        # Test cases
        lists1 = [[1,4,5],[1,3,4],[2,6]]
        lists2 = [[]]
        lists3 = []
        
        print(mergeKLists(lists1))  # Output: [1,1,2,3,4,4,5,6]
        print(mergeKLists(lists2))  # Output: []
        print(mergeKLists(lists3))  # Output: []
        """,
        """
        def isMatch(s: str, p: str) -> bool:
            m, n = len(s), len(p)
            dp = [[False] * (n + 1) for _ in range(m + 1)]
            dp[0][0] = True
            
            for j in range(1, n + 1):
                if p[j - 1] == '*':
                    dp[0][j] = dp[0][j - 1]
            
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if p[j - 1] == '?' or s[i - 1] == p[j - 1]:
                        dp[i][j] = dp[i - 1][j - 1]
                    elif p[j - 1] == '*':
                        dp[i][j] = dp[i - 1][j] or dp[i][j - 1]
            
            return dp[m][n]
        
        # Test cases
        s1, p1 = "aa", "a"
        s2, p2 = "aa", "*"
        s3, p3 = "cb", "?a"
        
        print(isMatch(s1, p1))  # Output: False
        print(isMatch(s2, p2))  # Output: True
        print(isMatch(s3, p3))  # Output: False
        """,
        """
        def solveSudoku(board):
            def is_valid(row, col, num):
                for i in range(9):
                    if board[row][i] == num or board[i][col] == num or board[(row//3)*3 + i//3][(col//3)*3 + i%3] == num:
                        return False
                return True
            
            def backtrack():
                for i in range(9):
                    for j in range(9):
                        if board[i][j] == '.':
                            for num in map(str, range(1, 10)):
                                if is_valid(i, j, num):
                                    board[i][j] = num
                                    if backtrack():
                                        return True
                                    board[i][j] = '.'
                            return False
                return True
            
            backtrack()
        
        # Test case
        board = [["5","3",".",".","7",".",".",".","."],
                 ["6",".",".","1","9","5",".",".","."],
                 [".","9","8",".",".",".",".","6","."],
                 ["8",".",".",".","6",".",".",".","3"],
                 ["4",".",".","8",".","3",".",".","1"],
                 ["7",".",".",".","2",".",".",".","6"],
                 [".","6",".",".",".",".","2","8","."],
                 [".",".",".","4","1","9",".",".","5"],
                 [".",".",".",".","8",".",".","7","9"]]
        
        solveSudoku(board)
        print(board)
        """,
        """
        def fullJustify(words, maxWidth):
            result = []
            line = []
            line_length = 0
            for word in words:
                if line_length + len(word) + len(line) > maxWidth:
                    for i in range(maxWidth - line_length):
                        line[i % (len(line) - 1 or 1)] += ' '
                    result.append(''.join(line))
                    line = []
                    line_length = 0
                line.append(word)
                line_length += len(word)
            result.append(' '.join(line).ljust(maxWidth))
            return result
        
        # Test cases
        words1 = ["This", "is", "an", "example", "of", "text", "justification."]
        maxWidth1 = 16
        print(fullJustify(words1, maxWidth1))
        
        words2 = ["What","must","be","acknowledgment","shall","be"]
        maxWidth2 = 16
        print(fullJustify(words2, maxWidth2))
        
        words3 = ["Science","is","what","we","understand","well","enough","to","explain","to","a","computer.","Art","is","everything","else","we","do"]
        maxWidth3 = 20
        print(fullJustify(words3, maxWidth3))
        """,
        """
        from collections import deque
        
        def ladderLength(beginWord, endWord, wordList):
            wordSet = set(wordList)
            if endWord not in wordSet:
                return 0
            
            queue = deque([(beginWord, 1)])
            visited = set()
            
            while queue:
                word, level = queue.popleft()
                if word == endWord:
                    return level
                for i in range(len(word)):
                    for char in 'abcdefghijklmnopqrstuvwxyz':
                        new_word = word[:i] + char + word[i+1:]
                        if new_word in wordSet and new_word not in visited:
                            visited.add(new_word)
                            queue.append((new_word, level + 1))
            return 0
        
        # Test cases
        beginWord1, endWord1, wordList1 = "hit", "cog", ["hot","dot","dog","lot","log","cog"]
        print(ladderLength(beginWord1, endWord1, wordList1))  # Output: 5
        
        beginWord2, endWord2, wordList2 = "hit", "cog", ["hot","dot","dog","lot","log"]
        print(ladderLength(beginWord2, endWord2, wordList2))  # Output: 0
        """,
        """
        def addOperators(num, target):
            def generateExpressions(idx, expr, evaluated, target):
                if idx == len(num):
                    if evaluated == target:
                        result.append(expr)
                    return
                
                for i in range(idx, len(num)):
                    if i > idx and num[idx] == '0':  # Skip leading zeros
                        break
                        
                    curr_str = num[idx:i+1]
                    curr_val = int(curr_str)
                    
                    if idx == 0:
                        generateExpressions(i + 1, curr_str, curr_val, target)
                    else:
                        generateExpressions(i + 1, expr + '+' + curr_str, evaluated + curr_val, target)
                        generateExpressions(i + 1, expr + '-' + curr_str, evaluated - curr_val, target)
                        generateExpressions(i + 1, expr + '*' + curr_str, evaluated - prev_val + prev_val * curr_val, target)
                        
            result = []
            generateExpressions(0, "", 0, target)
            return result
        
        # Test cases
        num1, target1 = "123", 6
        print(addOperators(num1, target1))  # Output: ["1*2*3","1+2+3"]
        
        num2, target2 = "232", 8
        print(addOperators(num2, target2))  # Output: ["2*3+2","2+3*2"]
        
        num3, target3 = "3456237490", 9191
        print(addOperators(num3, target3))  # Output: []
        """,
        """
        def addOperators(num, target):
            def generateExpressions(idx, expr, evaluated, prev_val, target):
                if idx == len(num):
                    if evaluated == target:
                        result.append(expr)
                    return
                
                for i in range(idx, len(num)):
                    if i > idx and num[idx] == '0':  # Skip leading zeros
                        break
                        
                    curr_str = num[idx:i+1]
                    curr_val = int(curr_str)
                    
                    if idx == 0:
                        generateExpressions(i + 1, curr_str, curr_val, curr_val, target)
                    else:
                        generateExpressions(i + 1, expr + '+' + curr_str, evaluated + curr_val, curr_val, target)
                        generateExpressions(i + 1, expr + '-' + curr_str, evaluated - curr_val, -curr_val, target)
                        generateExpressions(i + 1, expr + '*' + curr_str, evaluated - prev_val + prev_val * curr_val, prev_val * curr_val, target)
                        
            result = []
            generateExpressions(0, "", 0, 0, target)
            return result
        
        # Test cases
        num1, target1 = "123", 6
        print(addOperators(num1, target1))  # Output: ["1*2*3","1+2+3"]
        
        num2, target2 = "232", 8
        print(addOperators(num2, target2))  # Output: ["2*3+2","2+3*2"]
        
        num3, target3 = "3456237490", 9191
        print(addOperators(num3, target3))  # Output: []
        """,
        """
        class TreeNode:
            def __init__(self, val=0, left=None, right=None):
                self.val = val
                self.left = left
                self.right = right
        
        class Codec:
            def serialize(self, root):
                def serializeHelper(node):
                    if not node:
                        return "null"
                    return str(node.val) + "," + serializeHelper(node.left) + "," + serializeHelper(node.right)
                
                return serializeHelper(root)
            
            def deserialize(self, data):
                def deserializeHelper(queue):
                    if queue[0] == "null":
                        queue.popleft()
                        return None
                    
                    val = int(queue.popleft())
                    node = TreeNode(val)
                    node.left = deserializeHelper(queue)
                    node.right = deserializeHelper(queue)
                    return node
                
                data = data.split(",")
                return deserializeHelper(deque(data))
        
        # Test case
        root1 = TreeNode(1)
        root1.left = TreeNode(2)
        root1.right = TreeNode(3)
        root1.right.left = TreeNode(4)
        root1.right.right = TreeNode(5)
        
        codec = Codec()
        serialized_tree = codec.serialize(root1)
        print("Serialized tree:", serialized_tree)  # Output: "1,2,null,null,3,4,null,null,5,null,null"
        
        deserialized_tree = codec.deserialize(serialized_tree)
        print("Deserialized tree:", deserialized_tree)  # Output: Node object of the deserialized tree
        """,
        """
        class Solution:
            def numberToWords(self, num: int) -> str:
                if num == 0:
                    return "Zero"
                
                def toWords(n):
                    below_20 = ["", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten", "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen", "Seventeen", "Eighteen", "Nineteen"]
                    tens = ["", "", "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety"]
                    thousand_units = ["", "Thousand", "Million", "Billion"]
                    
                    if n == 0:
                        return ""
                    elif n < 20:
                        return below_20[n] + " "
                    elif n < 100:
                        return tens[n // 10] + " " + toWords(n % 10)
                    else:
                        return below_20[n // 100] + " Hundred " + toWords(n % 100)
                
                result = ""
                for i in range(len(thousand_units)):
                    if num % 1000 != 0:
                        result = toWords(num % 1000) + thousand_units[i] + " " + result
                    num //= 1000
                
                return result.strip()
        """,
        """
        import heapq
        
        class MedianFinder:
        
            def __init__(self):
                self.min_heap = []  # Stores the larger half of the numbers
                self.max_heap = []  # Stores the smaller half of the numbers
        
            def addNum(self, num: int) -> None:
                # Add the number to the appropriate heap
                if not self.max_heap or num <= -self.max_heap[0]:
                    heapq.heappush(self.max_heap, -num)
                else:
                    heapq.heappush(self.min_heap, num)
                
                # Balance the heaps if necessary
                if len(self.max_heap) > len(self.min_heap) + 1:
                    heapq.heappush(self.min_heap, -heapq.heappop(self.max_heap))
                elif len(self.min_heap) > len(self.max_heap):
                    heapq.heappush(self.max_heap, -heapq.heappop(self.min_heap))
        
            def findMedian(self) -> float:
                # If the number of elements is even, return the average of the two middle elements
                if len(self.max_heap) == len(self.min_heap):
                    return (-self.max_heap[0] + self.min_heap[0]) / 2
                else:
                    return -self.max_heap[0]  # If the number of elements is odd, return the middle element
        
        # Example usage:
        medianFinder = MedianFinder()
        medianFinder.addNum(1)
        medianFinder.addNum(2)
        print(medianFinder.findMedian())  # Output: 1.5
        medianFinder.addNum(3)
        print(medianFinder.findMedian())  # Output: 2.0
        """,
        """
        class TreeNode:
            def __init__(self, val=0, left=None, right=None):
                self.val = val
                self.left = left
                self.right = right
        
        class Codec:
        
            def serialize(self, root: TreeNode) -> str:
                ""Encodes a tree to a single string.""
                def preorder(node):
                    if not node:
                        return ["null"]
                    return [str(node.val)] + preorder(node.left) + preorder(node.right)
                
                return ",".join(preorder(root))
        
            def deserialize(self, data: str) -> TreeNode:
                "Decodes your encoded data to tree."
                def build_tree(nodes):
                    val = nodes.pop(0)
                    if val == "null":
                        return None
                    node = TreeNode(int(val))
                    node.left = build_tree(nodes)
                    node.right = build_tree(nodes)
                    return node
                
                nodes = data.split(",")
                return build_tree(nodes)
        
        # Example usage:
        # root = TreeNode(1)
        # root.left = TreeNode(2)
        # root.right = TreeNode(3)
        # root.right.left = TreeNode(4)
        # root.right.right = TreeNode(5)
        # codec = Codec()
        # serialized_tree = codec.serialize(root)
        # print(serialized_tree)  # Output: "1,2,null,null,3,4,null,null,5,null,null"
        # deserialized_tree = codec.deserialize(serialized_tree)
        """,
        """
        from collections import deque
        
        class Solution:
            def removeInvalidParentheses(self, s: str) -> List[str]:
                def is_valid(s):
                    count = 0
                    for ch in s:
                        if ch == '(':
                            count += 1
                        elif ch == ')':
                            count -= 1
                            if count < 0:
                                return False
                    return count == 0
        
                result = set()
                queue = deque([s])
                found_valid = False
                
                while queue:
                    curr = queue.popleft()
                    if is_valid(curr):
                        result.add(curr)
                        found_valid = True
                    if found_valid:
                        continue
                    for i in range(len(curr)):
                        if curr[i] in '()':
                            new_str = curr[:i] + curr[i+1:]
                            queue.append(new_str)
                
                return list(result)
        """,
        """
        class Solution:
            def maxCoins(self, nums: List[int]) -> int:
                nums = [1] + nums + [1]
                n = len(nums)
                dp = [[0] * n for _ in range(n)]
                
                for length in range(1, n):
                    for i in range(n - length):
                        j = i + length - 1
                        for k in range(i, j + 1):
                            coins = nums[i] * nums[k] * nums[j + 1]  # Coins from bursting balloon at position k
                            coins += dp[i][k - 1] + dp[k + 1][j]  # Coins from left and right subproblems
                            dp[i][j] = max(dp[i][j], coins)
                
                return dp[0][n - 1]
        """,
        """
        class Solution:
            def countSmaller(self, nums: List[int]) -> List[int]:
                def mergeSort(nums, indices):
                    if len(nums) <= 1:
                        return nums, [0]  # Base case: a single element has 0 smaller elements to the right
                    
                    mid = len(nums) // 2
                    left, left_indices = mergeSort(nums[:mid], indices[:mid])
                    right, right_indices = mergeSort(nums[mid:], indices[mid:])
                    
                    merged = []
                    merged_indices = []
                    count = 0
                    
                    i, j = 0, 0
                    while i < len(left) or j < len(right):
                        if j == len(right) or (i < len(left) and left[i] <= right[j]):
                            merged.append(left[i])
                            merged_indices.append(left_indices[i])
                            result[left_indices[i]] += count  # Update the count of smaller elements
                            i += 1
                        else:
                            merged.append(right[j])
                            merged_indices.append(right_indices[j])
                            count += 1  # Increment count for each element from the right half
                            j += 1
                    
                    return merged, merged_indices
                
                result = [0] * len(nums)
                mergeSort(nums, list(range(len(nums))))
                return result
        """,
        """
        class Solution:
            def maxNumber(self, nums1: List[int], nums2: List[int], k: int) -> List[int]:
                def maxNumberFromOneArray(nums, k):
                    stack = []
                    drop = len(nums) - k
                    for num in nums:
                        while drop and stack and stack[-1] < num:
                            stack.pop()
                            drop -= 1
                        stack.append(num)
                    return stack[:k]
                
                def merge(nums1, nums2):
                    merged = []
                    while nums1 or nums2:
                        if nums1 > nums2:
                            merged.append(nums1.pop(0))
                        else:
                            merged.append(nums2.pop(0))
                    return merged
                
                result = []
                for i in range(max(0, k - len(nums2)), min(k, len(nums1)) + 1):
                    maxNum1 = maxNumberFromOneArray(nums1, i)
                    maxNum2 = maxNumberFromOneArray(nums2, k - i)
                    merged = merge(maxNum1, maxNum2)
                    result = max(result, merged)
                return result
        """,
        """
        class Solution:
            def countRangeSum(self, nums: List[int], lower: int, upper: int) -> int:
                def mergeSort(prefixSums, start, end):
                    if end - start <= 1:
                        return 0
                    
                    mid = (start + end) // 2
                    count = mergeSort(prefixSums, start, mid) + mergeSort(prefixSums, mid, end)
                    
                    j = k = t = mid
                    temp = [0] * (end - start)
                    for i in range(start, mid):
                        while k < end and prefixSums[k] - prefixSums[i] < lower:
                            k += 1
                        while j < end and prefixSums[j] - prefixSums[i] <= upper:
                            j += 1
                        while t < end and prefixSums[t] < prefixSums[i]:
                            temp[t - start] = prefixSums[t]
                            t += 1
                        temp[i - start] = prefixSums[i]
                        count += j - k
                    prefixSums[start:t] = temp[:t - start]
                    return count
                
                prefixSums = [0] * (len(nums) + 1)
                for i in range(len(nums)):
                    prefixSums[i + 1] = prefixSums[i] + nums[i]
                
                return mergeSort(prefixSums, 0, len(prefixSums))
        """,
        """
        class Solution:
            def longestIncreasingPath(self, matrix: List[List[int]]) -> int:
                if not matrix:
                    return 0
                
                m, n = len(matrix), len(matrix[0])
                memo = [[None] * n for _ in range(m)]
                max_length = 0
                
                def dfs(i, j):
                    if memo[i][j] is not None:
                        return memo[i][j]
                    
                    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
                    max_path = 1
                    
                    for dx, dy in directions:
                        x, y = i + dx, j + dy
                        if 0 <= x < m and 0 <= y < n and matrix[x][y] > matrix[i][j]:
                            max_path = max(max_path, 1 + dfs(x, y))
                    
                    memo[i][j] = max_path
                    return max_path
                
                for i in range(m):
                    for j in range(n):
                        max_length = max(max_length, dfs(i, j))
                
                return max_length
        """,
        """
        class Solution:
            def minPatches(self, nums: List[int], n: int) -> int:
                miss = 1
                added = 0
                i = 0
                
                while miss <= n:
                    if i < len(nums) and nums[i] <= miss:
                        miss += nums[i]
                        i += 1
                    else:
                        miss *= 2
                        added += 1
                
                return added
        """,
        """
        from collections import defaultdict

        class Solution:
            def findItinerary(self, tickets: List[List[str]]) -> List[str]:
                graph = defaultdict(list)
                
                # Step 1: Create the graph
                for ticket in sorted(tickets, reverse=True):
                    graph[ticket[0]].append(ticket[1])
                
                # Step 2: Implement DFS
                def dfs(curr):
                    while graph[curr]:
                        dfs(graph[curr].pop())
                    route.append(curr)
                
                # Initialize the route with the starting airport "JFK"
                route = []
                dfs("JFK")
                
                # Reverse the route to get the correct order
                return route[::-1]
        """,
        """
        class Solution:
            def isPathCrossing(self, distance: List[int]) -> bool:
                # Initialize position and visited set
                x, y = 0, 0
                visited = {(0, 0)}
                
                # Define directions: north, west, south, east
                directions = [(0, 1), (-1, 0), (0, -1), (1, 0)]
                direction_index = 0
                
                # Traverse the distance array
                for d in distance:
                    dx, dy = directions[direction_index]
                    # Update position
                    for _ in range(d):
                        x += dx
                        y += dy
                        # Check if current position is visited
                        if (x, y) in visited:
                            return True
                        visited.add((x, y))
                    # Update direction
                    direction_index = (direction_index + 1) % 4
                
                return False
        """,
        """
        class Solution:
            def palindromePairs(self, words: List[str]) -> List[List[int]]:
                # Initialize hash table to store indices of words
                word_indices = {word: i for i, word in enumerate(words)}
                result = []
                
                # Helper function to check if a string is palindrome
                def is_palindrome(s):
                    return s == s[::-1]
                
                # Iterate over words
                for i, word in enumerate(words):
                    # Check if the word itself is palindrome and its reverse exists in the hash table
                    if word and is_palindrome(word) and "" in word_indices:
                        j = word_indices[""]
                        if i != j:
                            result.append([i, j])
                            result.append([j, i])
                    
                    # Check if the word can form palindrome pairs with its prefix or suffix
                    for j in range(len(word)):
                        prefix = word[:j]
                        suffix = word[j:]
                        if is_palindrome(prefix):
                            reverse_suffix = suffix[::-1]
                            if reverse_suffix in word_indices and word_indices[reverse_suffix] != i:
                                result.append([i, word_indices[reverse_suffix]])
                        if is_palindrome(suffix):
                            reverse_prefix = prefix[::-1]
                            if reverse_prefix in word_indices and word_indices[reverse_prefix] != i:
                                result.append([word_indices[reverse_prefix], i])
                
                return result
        """,
        """
        class SummaryRanges:

            def __init__(self):
                self.intervals = []
        
            def addNum(self, val: int) -> None:
                # Convert value to an interval
                interval = [val, val]
                
                # Merge the new interval with existing ones
                merged = []
                for start, end in self.intervals:
                    if interval[1] + 1 < start:  # If interval ends before current interval starts
                        merged.append(interval)
                        interval = [start, end]
                    elif interval[0] - 1 > end:  # If interval starts after current interval ends
                        merged.append([start, end])
                    else:  # If there's overlap, merge intervals
                        interval[0] = min(interval[0], start)
                        interval[1] = max(interval[1], end)
                
                merged.append(interval)
                self.intervals = merged
        
            def getIntervals(self) -> List[List[int]]:
                return self.intervals
        """,
        """
        def maxEnvelopes(envelopes):
            # Sort envelopes based on widths in ascending order
            envelopes.sort(key=lambda x: (x[0], -x[1]))
            
            # Use dynamic programming to find the longest increasing subsequence of heights
            dp = []
            for _, h in envelopes:
                left, right = 0, len(dp)
                while left < right:
                    mid = left + (right - left) // 2
                    if dp[mid] < h:
                        left = mid + 1
                    else:
                        right = mid
                if right == len(dp):
                    dp.append(h)
                else:
                    dp[right] = h
            
            return len(dp)
        
        # Test cases
        print(maxEnvelopes([[5,4],[6,4],[6,7],[2,3]]))  # Output: 3
        print(maxEnvelopes([[1,1],[1,1],[1,1]]))        # Output: 1
        """,
        """
        import bisect

        def maxSumSubmatrix(matrix, k):
            m, n = len(matrix), len(matrix[0])
            max_sum = float('-inf')
        
            for left in range(n):
                prefix = [0] * m
                for right in range(left, n):
                    for i in range(m):
                        prefix[i] += matrix[i][right]
        
                    # Find the maximum subarray sum of prefix such that the sum is less than or equal to k
                    curr_sum = 0
                    prefix_sum = [0]
                    for num in prefix:
                        curr_sum += num
                        # Find the smallest prefix_sum such that curr_sum - prefix_sum <= k
                        idx = bisect.bisect_left(prefix_sum, curr_sum - k)
                        if idx < len(prefix_sum):
                            max_sum = max(max_sum, curr_sum - prefix_sum[idx])
                        # Insert curr_sum into prefix_sum array in sorted order
                        bisect.insort(prefix_sum, curr_sum)
        
            return max_sum
        
        # Test cases
        print(maxSumSubmatrix([[1,0,1],[0,-2,3]], 2))  # Output: 2
        print(maxSumSubmatrix([[2,2,-1]], 3))         # Output: 3
        """,
        """
        import random

        class RandomizedCollection:
        
            def __init__(self):
                self.val_to_indices = {}
                self.nums = []
        
            def insert(self, val: int) -> bool:
                self.nums.append(val)
                if val not in self.val_to_indices:
                    self.val_to_indices[val] = []
                self.val_to_indices[val].append(len(self.nums) - 1)
                return len(self.val_to_indices[val]) == 1
        
            def remove(self, val: int) -> bool:
                if val not in self.val_to_indices:
                    return False
                idx = self.val_to_indices[val].pop()
                last_val = self.nums[-1]
                self.nums[idx] = last_val
                self.val_to_indices[last_val].remove(len(self.nums) - 1)
                if idx < len(self.nums) - 1:
                    self.val_to_indices[last_val].append(idx)
                if not self.val_to_indices[val]:
                    del self.val_to_indices[val]
                self.nums.pop()
                return True
        
            def getRandom(self) -> int:
                return random.choice(self.nums)
        """,
        """
        import random

        class RandomizedCollection:
        
            def __init__(self):
                self.val_to_indices = {}
                self.nums = []
        
            def insert(self, val: int) -> bool:
                self.nums.append(val)
                if val not in self.val_to_indices:
                    self.val_to_indices[val] = set()
                self.val_to_indices[val].add(len(self.nums) - 1)
                return len(self.val_to_indices[val]) == 1
        
            def remove(self, val: int) -> bool:
                if val not in self.val_to_indices:
                    return False
                idx = self.val_to_indices[val].pop()
                last_val = self.nums[-1]
                self.nums[idx] = last_val
                self.val_to_indices[last_val].remove(len(self.nums) - 1)
                if idx < len(self.nums) - 1:
                    self.val_to_indices[last_val].add(idx)
                if not self.val_to_indices[val]:
                    del self.val_to_indices[val]
                self.nums.pop()
                return True
        
            def getRandom(self) -> int:
                return random.choice(self.nums)
        """
    ]

    with open('chatgpt_code.txt', 'w', encoding='utf-8') as file:
        for code in chatgpt_code_examples:
            file.write(code.strip() + '\n\n')


def save_human_code(token, repo_name, max_files=1000):
    g = Github(token)
    repo = g.get_repo(repo_name)
    contents = []

    def fetch_contents(path=""):
        try:
            contents_page = repo.get_contents(path)
            if not isinstance(contents_page, list):
                contents_page = [contents_page]

            while contents_page and len(contents) < max_files:  # 특정 파일 개수까지만 가져오도록 제한
                file_content = contents_page.pop(0)
                if file_content.type == 'dir':
                    contents_page.extend(repo.get_contents(file_content.path))
                else:
                    contents.append(file_content)

                # 일정 시간 간격으로 대기
                time.sleep(1)  # 예시로 1초 대기

        except GithubException as e:
            if e.status == 403 and 'rate limit exceeded' in str(e.data).lower():
                rate_limit = g.get_rate_limit().core
                reset_timestamp = rate_limit.reset.timestamp()
                sleep_time = reset_timestamp - time.time() + 1  # Adding 1 second buffer
                print(f"Rate limit exceeded. Sleeping for {sleep_time} seconds.")
                time.sleep(sleep_time)
                fetch_contents(path)
            else:
                print(f"Error fetching contents: {e}")
                raise e

    fetch_contents()
    print(f"Collected {len(contents)} files.")
    return contents

def merge_human_code():
    human_code_directory = 'human_code'
    human_code_files = [os.path.join(human_code_directory, file) for file in os.listdir(human_code_directory) if file.endswith('.py')]

    human_code_examples = []
    for file_path in human_code_files:
        with open(file_path, 'r', encoding='utf-8') as file:
            human_code_examples.append(file.read().strip())

    with open('human_code.txt', 'w', encoding='utf-8') as file:
        for code in human_code_examples:
            file.write(code + '\n\n')

# 다른 코드에서는 호출되지 않았지만, 이 코드가 직접 실행되는 경우에만 실행됩니다.
if __name__ == "__main__":
    # 사용 예시
    save_human_code('your_github_token', 'owner/repository')
    merge_human_code()
