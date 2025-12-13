A = [19, 5, 2, 7, 29, 3, 4, 12, 6]

def bubble_sor(nums):
    n = len(nums)
    for i in range(n):
        for j in range(n-i-1):
            if nums[j] > nums[j+1]:
                nums[j], nums[j+1] = nums[j+1], nums[j]

def select_sort(nums):
    n = len(nums)
    for i in range(n):
        min_i = i
        for j in range(i+1, n):
            if nums[j] < nums[min_i]:
                min_i = j
        nums[i], nums[min_i] = nums[min_i], nums[i]

def insertion_sort(nums):
    n = len(nums)
    for i in range(1, n):
        cur = nums[i]
        j = i - 1
        while j >= 0 and nums[j] > cur:
            nums[j+1] = nums[j]
            j -= 1
        nums[j+1] = cur

def quick_sort(nums):
    if len(nums) <= 1:
        return nums
    pivot = nums[0]
    left = [num for num in nums[1:] if num <= pivot]
    right = [num for num in nums[1:] if num > pivot]
    return quick_sort(left) + [pivot] + quick_sort(right)

def merge(l, r):
    ans = []
    i = j = 0
    while i < len(l) and j < len(r):
        if l[i] > r[j]:
            ans.append(r[j])
            j += 1
        else:
            ans.append(l[i])
            i += 1
    ans.extend(l[i:])
    ans.extend(r[j:])
    return ans

def merge_sort(nums):
    if len(nums) <= 1:
        return nums
    n = len(nums)
    mid = n // 2
    left = merge_sort(nums[:mid])
    right = merge_sort(nums[mid:])
    return merge(left, right)

def heap_sort(nums):
    import heapq
    heapq.heapify(nums)
    return [heapq.heappop(nums) for _ in range(len(nums))]

if __name__ == "__main__":
    # bubble_sor(A)
    # B = quick_sort(A)
    # select_sort(A)
    # insertion_sort(A)
    # B = merge_sort(A)
    B = heap_sort(A)
    print(B)
