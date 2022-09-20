class Tree:
    def __init__(self):
        self.feature_value = None
        self.threshold = None
        self.left_child = None
        self.right_child = None

    def set_children(self, left_child, right_child):
        self.left_child = left_child
        self.right_child = right_child

    def set_threshold(self, threshold):
        self.threshold = self.threshold 

# class Tree:
#     def __init__(self, root):
#         if isinstance(root,Node):
#             self.root = root
#         else:
#             raise ValueError("")
