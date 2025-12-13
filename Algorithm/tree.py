from typing import Optional, List
from collections import deque

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

    # left -> mid -> right
    def inorder(self, root: Optional['TreeNode']) -> List[int]:
        WHITE, GRAY = 0, 1
        inorder = []
        stack = [(root, WHITE)]
        while stack:
            node, state = stack.pop()
            if node is None: continue
            if state == WHITE:
                stack.append((node.right, WHITE))
                stack.append((node, GRAY))
                stack.append((node.left, WHITE))
            else:
                inorder.append(node.val)
        return inorder
    # mid -> left -> right
    def preorder(self, root: Optional['TreeNode']) -> List[int]:
        WHITE, GRAY = 0, 1
        preorder = []
        stack = [(root, WHITE)]
        while stack:
            node, state = stack.pop()
            if node is None: continue
            if state == WHITE:
                stack.append((node.right, WHITE))
                stack.append((node.left, WHITE))
                stack.append((node, GRAY))
            else:
                preorder.append(node.val)
        return preorder
    # left -> right -> mid
    def postorder(self, root: Optional['TreeNode']) -> List[int]:
        WHITE, GRAY = 0, 1
        postorder = []
        stack = [(root, WHITE)]
        while stack:
            node, state = stack.pop()
            if node is None: continue
            if state == WHITE:
                stack.append((node, GRAY))
                stack.append((node.right, WHITE))
                stack.append((node.left, WHITE))
            else:
                postorder.append(node.val)
        return postorder

def build_tree(level: List[Optional[int]]) -> Optional[TreeNode]:
    if not level:
        return None

    root = TreeNode(level[0])
    queue = deque([root])
    i = 1

    while queue and i < len(level):
        node = queue.popleft()

        if i < len(level) and level[i] is not None:
            node.left = TreeNode(level[i])
            queue.append(node.left)
        i += 1

        if i < len(level) and level[i] is not None:
            node.right = TreeNode(level[i])
            queue.append(node.right)
        i += 1

    return root 

if __name__ == "__main__":
    root = build_tree([1, 2, 3, 4, 5, None, None])
    # [4, 2, 5, 1, 3]
    print(root.inorder(root))
    # [1, 2, 4, 5, 3]
    print(root.preorder(root))
    # [4, 5, 2, 3, 1]
    print(root.postorder(root))   