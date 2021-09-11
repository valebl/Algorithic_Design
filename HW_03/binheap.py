from typing import TypeVar, Generic, Union, List
from numbers import Number

T = TypeVar('T')

def min_order(a: Number, b: Number) -> bool:
    return a <= b
    
def max_order(a: Number, b: Number) -> bool:
    return a >= b
    
class binheap(Generic[T]):
    LEFT = 0
    RIGHT = 1
    
    def __init__(self, A: Union[int, List[T]], total_order=None):
        
        if total_order is None:
            self._torder = min_order
        else:
            self._torder = total_order
        
        if isinstance(A, int):
            self._size = 0 # empty heap
            self._A = [None] * A # in the future the heap will be able to store up to A values
            self._V = [] # array which stores the structure of the heap    
        else:
            self._size = len(A)
            self._A = A # array which will store all keys in our heap 
            self._V = [i for i in range(self._size)] # array which stores the structure of the heap           
        
        self._build_heap()
    
    @staticmethod
    def parent(node: int) -> Union[int, None]:
        if node == 0: # if the node is the root there is no parent
            return None
        return (node-1)//2

    @staticmethod
    def child(node: int, side: int) -> int:
        return 2*node + 1 + side # depending on side we return left or right child
         
    @staticmethod
    def left(node: int) -> int:
        return 2*node + 1
        
    @staticmethod
    def right(node: int) -> int:
        return 2*node + 2
        
    def __len__(self):
        return self._size
        
    def is_empty(self) -> bool:
        return self._size == 0    
        
    def _swap_keys(self, node_a: int, node_b: int) -> None:
        # swap the nodes in the array representation
        self._A[node_a], self._A[node_b] = self._A[node_b], self._A[node_a]
        # swap the positions in the list representing the structure of the heap
        self._V[self._A[node_a][0]] = node_a
        self._V[self._A[node_b][0]] = node_b

    def _heapify(self, node: int) -> None: 
        keep_fixing = True # to decide whether the heapify function should keep going
        
        while keep_fixing: # we implement an iterative version
            min_node = node
            for child_idx in [binheap.left(node), binheap.right(node)]:
                if (child_idx < self._size and
                        self._torder(self._A[child_idx], self._A[min_node])): # check if valid index
                    min_node = child_idx
                    # min_node is the index of the minimum key among the keys of root and its children
                    
            if min_node != node:
                self._swap_keys(min_node,node)
                node = min_node
            else:
                keep_fixing = False
                    
    def remove_minimum(self) -> T:
        if self.is_empty():
            raise RuntimeError('The heap is empty - cannot remove the minimum')
            
        self._swap_keys(0,self._size-1)
        self._V[self._A[self._size-1][0]] = None       
        self._size = self._size-1        
        self._heapify(0) # by calling heapify on he root we fix all the heap
        
        return self._A[self._size]
        
    def _build_heap(self) -> None:
        for i in range(binheap.parent(self._size-1),-1,-1): # to avoid n/2 calls of heapify
            self._heapify(i)
            
    def decrease_key(self, vertex: int, new_value: T) -> None:
        node = self._V[vertex]
        if self._torder(self._A[node], new_value):
            raise RuntimeError(f'{new_value} is not smaller than {self._A[node]}')

        self._A[node] = new_value
        parent = binheap.parent(node)
        while (node != 0 and not self._torder(self._A[parent],
                                                self._A[node])):
            self._swap_keys(node, parent)            
            node = parent
            parent = binheap.parent(node)
            
    def insert(self, value: T) -> None:
        if self._size >= len(self._A):
            raise RuntimeError('The heap is full')
            
        if self.is_empty():
            self._A[0] = value
            self._V[value[0]] = 0
            self._size += 1
        else:
            if self._torder(self._A[binheap.parent(self._size)], value):
                self._A[self._size] = value # the heap property is already satisfied
                self._V[value[0]] = self._size
                self._size += 1
            else:
                self._A[self._size] = self._A[binheap.parent(self._size)]
                self._V[value[0]] = self._size
                self._size += 1
                self.decrease_key(self._size - 1, value)
        
    
    def __repr__(self) -> str:
        bh_str = ''
        
        next_node = 1
        up_to = 2
        
        while next_node <= self._size:
            level = '\t'.join(f'{v}' for v in self._A[next_node-1: min(up_to-1, self._size)])
            
            if next_node == 1:
                bh_str = level
            else:
                bh_str += f'\n{level}'
                
            next_node = up_to
            up_to = 2*up_to
            
        return bh_str

    def __iter__(self):
        return iter(self._A)
        




