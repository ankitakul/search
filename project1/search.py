# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to Clemson University and the authors.
# 
# Authors: Pei Xu (peix@g.clemson.edu) and Ioannis Karamouzas (ioannis@g.clemson.edu)
#

"""
 In this assignment, the task is to implement different search algorithms to find 
 a path from a given start cell to the goal cell in a 2D grid map.

 To complete the assignment, you must finish three functions:
   depth_first_search (line 148), uniform_cost_search (line 224), 
   and astar_search (line 264).

 During the search, a cell is represented using a tuple of its location
 coordinates.
 For example:
   the cell at the left-top corner is (0, 0);
   the cell at the first row and second column is (0, 1);
   the cell at the second row and first column is (1, 0).
 You need put these tuples into the open set or/and closed set properly
 during searching.
"""

# ACTIONS defines how to reach an adjacent cell from a given cell
# Important: please check that a cell within the bounds of the grid
# when try to access it.
ACTIONS = (
    (-1,  0), # go up
    ( 0, -1), # go left
    ( 1,  0), # go down
    ( 0,  1)  # go right
)

from utils.search_app import OrderedSet, Stack, Queue, PriorityQueue
"""
 Four different structures are provided as the containers of the open set
 and closed set.

 OrderedSet is an ordered collections of unique elements.

 Stack is an LIFO container whose `pop()` method always pops out the last
 added element. 

 Queue is an FIFO container whose `pop()` method always pops out the first
 added element.

 PriorityQueue is a key-value container whose `pop()` method always pops out
 the element whose value has the highest priority.

 All of these containers are iterable but not all of them are ordered. Use
 their pop() methods to ensure elements are popped out as expected.


 Common operations of OrderedSet, Stack, Queue, PriorityQueue
   len(s): number of elements in the container s
   x in s: test x for membership in s
   x not in s: text x for non-membership in s
   s.clear(): clear s
   s.remove(x): remove the element x from the set s;
                nothing will be done if x is not in s


 Unique operations of OrderedSet:
   s.add(x): add the element x into the set s;
             nothing will be done if x is already in s
   s.pop(): return and remove the LAST added element in s;
            raise IndexError if s is empty 
   s.pop(last=False): return and remove the FIRST added element in s;
            raise IndexError if s is empty 
 Example:
   s = Set()
   s.add((1,2))    # add a tuple element (1,2) into the set
   s.remove((1,2)) # remove the tuple element (1,2) from the set
   s.add((1,1))
   s.add((2,2))
   s.add((3,3))
   x = s.pop()
   assert(x == (3,3))
   assert((1,1) in s and (2,2) in s)
   assert((3,3) not in s)
   x = s.pop(last=False)
   assert(x == (1,1))
   assert((2,2) in s)
   assert((1,1) not in s)
   

 Unique operations of Stack:
   s.append(x): add the element x into the back of the stack s
   s.pop(): return and remove the LAST added element in the stack s;
            raise IndexError if s is empty
 Example:
   s = Stack()
   s.add((1,1))
   s.add((2,2))
   x = s.pop()
   assert(x == (2,2))
   assert((1,1) in s)
   assert((2,2) not in s)


 Unique operations of Queue:
   s.append(x): add the element x into the back of the queue s
   s.pop(): return and remove the FIRST added element in the queue s;
            raise IndexError if s is empty
 Example:
   s = Queue()
   s.add((1,1))
   s.add((2,2))
   x = s.pop()
   assert(x == (1,1))
   assert((2,2) in s)
   assert((1,1) not in s)


 Unique operations of PriorityQueue:
   PriorityQueue(order="min", f=lambda v: v): build up a priority queue
       using the function f to compute the priority based on the value
       of an element
   s.put(x, v): add the element x with value v into the queue
                update the value of x if x is already in the queue
   s.get(x): get the value of the element x
            raise KeyError if x is not in s
   s.pop(): return and remove the element with highest priority in s;
            raise IndexError if s is empty
            if order is "min", the element with minimum f(v) will be popped;
            if order is "max", the element with maximum f(v) will be popped.
 Example:
   s = PriorityQueue(order="max", f=lambda v: abs(v))
   s.put((1,1), -1)
   s.put((2,2), -20)
   s.put((3,3), 10)
   x, v = s.pop()  # the element with maximum value of abs(v) will be popped
   assert(x == (2,2) and v == -20)
   assert(x not in s)
   assert(x.get((1,1)) == -1)
   assert(x.get((3,3)) == 10)
"""


# use math library if needed
import math

def depth_first_search(grid_size, start, goal, obstacles, costFn, logger):
    """
    DFS algorithm finds the path from the start cell to the
    goal cell in the 2D grid world.
    
    Parameters
    ----------
    grid_size: tuple, (n_rows, n_cols)
        (number of rows of the grid, number of cols of the grid)
    start: tuple, (row, col)
        location of the start cell;
        row and col are counted from 0, i.e. the 1st row is 0
    goal: tuple, (row, col)
        location of the goal cell
    obstacles: tuple, ((row, col), (row, col), ...)
        locations of obstacles in the grid
        the cells where obstacles are located are not allowed to access 
    costFn: a function that returns the cost of landing to a cell (x,y)
         after taking an action. The default cost is 1, regardless of the
         action and the landing cell, i.e. every time the agent moves NSEW
         it costs one unit. 
    logger: a logger to visualize the search process.
         Do not do anything to it.

   

    Returns
    -------
    movement along the path from the start to goal cell: list of actions
        The first returned value is the movement list found by the search
        algorithm from the start cell to the end cell.
        The movement list should be a list object who is composed of actions
        that should made moving from the start to goal cell along the path
        found the algorithm.
        For example, if nodes in the path from the start to end cell are:
            (0, 0) (start) -> (0, 1) -> (1, 1) -> (1, 0) (goal)
        then, the returned movement list should be
            [(0,1), (1,0), (0, -1)]
        which means: move right, down, left.

        Return an EMPTY list if the search algorithm fails finding any
        available path.
        
    closed set: list of location tuple (row, col)
        The second returned value is the closed set, namely, the cells are expanded during search.
    """
    n_rows, n_cols = grid_size
    start_row, start_col = start
    goal_row, goal_col = goal

    ##########################################
    # Choose a proper container yourself from
    # OrderedSet, Stack, Queue, PriorityQueue
    # for the open set and closed set.
    open_set = Stack()
    closed_set = Stack()
    ##########################################

    ##########################################
    # Set up visualization logger hook
    # Please do not modify these four lines
    closed_set.logger = logger
    logger.closed_set = closed_set
    open_set.logger = logger
    logger.open_set = open_set
    ##########################################

    movement = []
    # ----------------------------------------
    # finish the code below
    # ----------------------------------------
#############################################################################
   
       
     
    def TransitionModel(node):
        current_row = node[0]
        current_col = node[1]
        direction_can_go = []
        
        TempRow =current_row
        TempCol =current_col+1
        TempNode =(TempRow,TempCol)
        
        if TempNode not in obstacles and TempRow >=0 and  TempCol >=0 and TempRow <= n_rows and TempCol <= n_cols:
            direction_can_go.append(TempNode)
        
       
        TempRow =current_row+1
        TempCol =current_col
        TempNode =(TempRow,TempCol)
        
        if TempNode not in obstacles and TempRow >=0 and  TempCol >=0 and TempRow <= n_rows and TempCol <= n_cols:
            direction_can_go.append(TempNode)
       
        TempRow =current_row-1
        TempCol =current_col
        TempNode =(TempRow,TempCol)
        
        if TempNode not in obstacles and TempRow >=0 and  TempCol >=0 and TempRow <= n_rows and TempCol <= n_cols:
            direction_can_go.append(TempNode)
        
        TempRow =current_row
        TempCol =current_col-1
        TempNode =(TempRow,TempCol)
        
        if TempNode not in obstacles and TempRow >=0 and  TempCol >=0 and TempRow <= n_rows and TempCol <= n_cols:
            direction_can_go.append(TempNode)
        #print("direction_can_go ::",direction_can_go)
        #print("node ::",node)
        return direction_can_go
      
    def ExpandNode():       
        while len(open_set):
            i=0
            node = open_set.pop()
            '''
            if i:
                prevnode=node   
                
                R= prevnode[0]-node[0]
                C =prevnode[1]-node[1]

                if(R == -1 and C == 0):
                    movement.append(ACTIONS[0])
                if(R == 0 and C == -1):
                    movement.append(ACTIONS[1])
                if(R == 1 and C == 0):
                    movement.append(ACTIONS[2])
                if(R == 0 and C == 1):
                    movement.append(ACTIONS[3])
            i=1
            '''
            closed_set.add(node)
            actions_posible = TransitionModel(node)
            
            for nextnode in actions_posible:
                #print("nextnode ::" ,nextnode)
                if nextnode not in closed_set and nextnode not in open_set:
                    if nextnode == goal:
                        closed_set.add(nextnode) 
                        print("found it")
                        movement = backtrace(parent, start, goal)
                        movement=closed_set
                        return   movement, closed_set
                    else:
                        open_set.add(nextnode)
                        #parent[nextnode] = node 
                        
                        R= node[0]-nextnode[0]
                        C =node[1]-nextnode[1]
                        
                        if(R == -1 and C == 0):
                            movement.append(ACTIONS[0])
                        if(R == 0 and C == -1):
                            movement.append(ACTIONS[1])
                        if(R == 1 and C == 0):
                            movement.append(ACTIONS[2])
                        if(R == 0 and C == 1):
                            movement.append(ACTIONS[3])
                        
                  
                    
    open_set.add(start)
    parent={}
    
    ExpandNode()
   
  
    
    
#############################################################################
    return movement, closed_set

def uniform_cost_search(grid_size, start, goal, obstacles, costFn, logger):
    """
    Uniform-cost search algorithm finds the optimal path from 
    the start cell to the goal cell in the 2D grid world. 
    
    After expanding a node, to retrieve the cost of a child node at location (x,y), 
    please call costFn((x,y)). In all of the grid maps, the cost is always 1.

    See depth_first_search() for details.
    """
    n_rows, n_cols = grid_size
    start_row, start_col = start
    goal_row, goal_col = goal

    ##########################################
    # Choose a proper container yourself from
    # OrderedSet, Stack, Queue, PriorityQueue
    # for the open set and closed set.
    open_set =  PriorityQueue(order="min", f=lambda v: abs(v))
    closed_set = Queue()
    ##########################################

    ##########################################
    # Set up visualization logger hook
    # Please do not modify these four lines
    closed_set.logger = logger
    logger.closed_set = closed_set
    open_set.logger = logger
    logger.open_set = open_set
    ##########################################

    movement = []
    # ----------------------------------------
    # finish the code below
    # ----------------------------------------
#############################################################################

    def backtrace(parent, start, goal):
        print("in backtrace")
        path = [goal]
        while path[-1] != start:
            path.append(parent[path[-1]])
        path.reverse()
        iter =len(path)       
        i = 0;
        while i < iter-1:
           # print(" path[i][0]" ,path[i][0])
           # print(" path[i+1][0]" ,path[i+1][0]) 
            R= path[i+1][0]-path[i][0]
            C =path[i+1][1]-path[i][1]           

            if(R == -1 and C == 0):
                movement.append(ACTIONS[0])
            if(R == 0 and C == -1):
                movement.append(ACTIONS[1])
            if(R == 1 and C == 0):
                movement.append(ACTIONS[2])
            if(R == 0 and C == 1):
                movement.append(ACTIONS[3])
            
            i += 1
            
        
        return movement, closed_set
   
    def TransitionModel(node, value):
        current_row = node[0]
        current_col = node[1]
        direction_can_go = []
        
           #down    
        TempRow =current_row
        TempCol =current_col+1
        TempNode =(TempRow,TempCol)
        
       # print("TempNode ::",TempNode)
        
        if TempNode not in obstacles and TempRow >=0 and  TempCol >=0 and TempRow <= n_rows and TempCol <= n_cols:
            returnnode= [TempNode, costFn(TempNode)]
            direction_can_go.append(returnnode)
        
        #right
        TempRow =current_row+1
        TempCol =current_col
        TempNode =(TempRow,TempCol)
        
        if TempNode not in obstacles and TempRow >=0 and  TempCol >=0 and TempRow <= n_rows and TempCol <= n_cols:
            returnnode= [TempNode, costFn(TempNode)]
            direction_can_go.append(returnnode)
        
        #left
        TempRow =current_row-1
        TempCol =current_col
        TempNode =(TempRow,TempCol)
        
        if TempNode not in obstacles and TempRow >=0 and  TempCol >=0 and TempRow <= n_rows and TempCol <= n_cols:
            returnnode= [TempNode,  costFn(TempNode)]
            direction_can_go.append(returnnode)
        #up
        TempRow =current_row
        TempCol =current_col-1
        TempNode =(TempRow,TempCol)
        
        if TempNode not in obstacles and TempRow >=0 and  TempCol >=0 and TempRow <= n_rows and TempCol <= n_cols:
            returnnode= [TempNode,  costFn(TempNode)]
            direction_can_go.append(returnnode)
       # print("direction_can_go ::",direction_can_go)
        #print("node ::",node)
        return direction_can_go
      
    def Gvalue(row,col):
        return abs(goal[0]-row + goal[1]-col)
                    
    open_set.put(start,0)
    parent = {}
    while len(open_set):
            node,value = open_set.pop()

            if node == goal:
                print("found it")
                movement, closed_set = backtrace(parent, start, goal)   
                print("movement ::" ,movement)
                return  movement, closed_set
            
            #costFn(node)
            closed_set.add(node) 
           
            actions_posible = TransitionModel(node,value)
           
            for nextnode in actions_posible:
                #print("nextnode ::" ,nextnode[0])
                if nextnode[0] not in closed_set and nextnode[0] not in open_set:                    
                    open_set.put(nextnode[0],nextnode[1])
                    parent[nextnode[0]] = node
                elif  nextnode[0] in open_set and nextnode[1] <  open_set.get(nextnode[0]):
                    open_set.put(nextnode[0],nextnode[1])
                    parent[nextnode[0]] = node       
    
    
#############################################################################
    return movement, closed_set

def astar_search(grid_size, start, goal, obstacles, costFn, logger):
    """
    A* search algorithm finds the optimal path from the start cell to the
    goal cell in the 2D grid world.

    After expanding a node, to retrieve the cost of a child node at location (x,y), 
    please call costFn((x,y)). In all of the grid maps, the cost is always 1.  

    See depth_first_search() for details.    
    """
    n_rows, n_cols = grid_size
    start_row, start_col = start
    goal_row, goal_col = goal

    ##########################################
    # Choose a proper container yourself from
    # OrderedSet, Stack, Queue, PriorityQueue
    # for the open set and closed set.
    open_set = PriorityQueue(order="min", f=lambda v: abs(v))
    closed_set = Queue()
    ##########################################

    ##########################################
    # Set up visualization logger hook
    # Please do not modify these four lines
    closed_set.logger = logger
    logger.closed_set = closed_set
    open_set.logger = logger
    logger.open_set = open_set
    ##########################################

    movement = []

    # ----------------------------------------
    # finish the code below to implement a Manhattan distance heuristic
    # ----------------------------------------
    
    #############################################################################
    def heuristic(row, col):
        R = goal[0]-row
        C =  goal[1]-col
        hyp = R*R + C*C        
        return abs(math.sqrt(hyp))
          
    def Gvalue(row,col):
        return abs(goal[0]-row + goal[1]-col)
    
    def backtrace(parent, start, goal):
       
        path = [goal]
        while path[-1] != start:           
            path.append(parent[path[-1]])
        path.reverse()
        
        iter =len(path)       
        i = 0;
        while i < iter-1:
          
            R= path[i+1][0]-path[i][0]
            C =path[i+1][1]-path[i][1]

            if(R == -1 and C == 0):
                movement.append(ACTIONS[0])
            if(R == 0 and C == -1):
                movement.append(ACTIONS[1])
            if(R == 1 and C == 0):
                movement.append(ACTIONS[2])
            if(R == 0 and C == 1):
                movement.append(ACTIONS[3])
         
            i += 1
        
       
        return movement, closed_set
   
    def TransitionModel(node, value):
        current_row = node[0]
        current_col = node[1]
        direction_can_go = []
        
           #down    
        TempRow =current_row
        TempCol =current_col+1
        TempNode =(TempRow,TempCol)
        
       # print("TempNode ::",TempNode)
        
        if TempNode not in obstacles and TempRow >=0 and  TempCol >=0 and TempRow <= n_rows and TempCol <= n_cols:
            returnnode= [TempNode, Gvalue(TempRow,TempCol)+ heuristic(TempRow,TempCol)]
            direction_can_go.append(returnnode)
        
        #right
        TempRow =current_row+1
        TempCol =current_col
        TempNode =(TempRow,TempCol)
        
        if TempNode not in obstacles and TempRow >=0 and  TempCol >=0 and TempRow <= n_rows and TempCol <= n_cols:
            returnnode= [TempNode,Gvalue(TempRow,TempCol)+ heuristic(TempRow,TempCol)]
            direction_can_go.append(returnnode)
        
        #left
        TempRow =current_row-1
        TempCol =current_col
        TempNode =(TempRow,TempCol)
        
        if TempNode not in obstacles and TempRow >=0 and  TempCol >=0 and TempRow <= n_rows and TempCol <= n_cols:
            returnnode= [TempNode, Gvalue(TempRow,TempCol)+ heuristic(TempRow,TempCol)]
            direction_can_go.append(returnnode)
        #up
        TempRow =current_row
        TempCol =current_col-1
        TempNode =(TempRow,TempCol)
        
        if TempNode not in obstacles and TempRow >=0 and  TempCol >=0 and TempRow <= n_rows and TempCol <= n_cols:
            returnnode= [TempNode, Gvalue(TempRow,TempCol)+ heuristic(TempRow,TempCol)]
            direction_can_go.append(returnnode)
       # print("direction_can_go ::",direction_can_go)
        #print("node ::",node)
        return direction_can_go

                    
    open_set.put(start,0)
    parent = {}
    while len(open_set):
            node,value = open_set.pop()

            if node == goal:
                print("found it")
                movement, closed_set = backtrace(parent, start, goal)   
                print("movement ::" ,movement)
                return  movement, closed_set
            
            #costFn(node)
            closed_set.add(node) 
           
            actions_posible = TransitionModel(node,value)
           
            for nextnode in actions_posible:
                #print("nextnode ::" ,nextnode[0])
                if nextnode[0] not in closed_set and nextnode[0] not in open_set:                    
                    open_set.put(nextnode[0],nextnode[1])
                    parent[nextnode[0]] = node
                elif  nextnode[0] in open_set and nextnode[1] <  open_set.get(nextnode[0]):
                    open_set.put(nextnode[0],nextnode[1])
                    parent[nextnode[0]] = node
#############################################################################
    return movement, closed_set

if __name__ == "__main__":
    # make sure actions and cost are defined correctly
    from utils.search_app import App
    assert(ACTIONS == App.ACTIONS)

    import tkinter as tk

    algs = {
        "Depth-First": depth_first_search,
        "Uniform Cost Search": uniform_cost_search,
        "A*": astar_search
    }

    root = tk.Tk()
    App(algs, root)
    root.mainloop()
    
 
 
