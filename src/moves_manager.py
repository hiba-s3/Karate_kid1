import queue

moves_queue = queue.Queue(maxsize=10)
def saveMove(move):
    """
    Add a move ("left" or "right") to the queue. 
    If the queue is full, it will not add the move.
    """
    if move not in ["left", "right"]:
        print(f"Error: Invalid move '{move}'")
        return
    if not moves_queue.full():
        moves_queue.put(move)
        
    else:
        print("Queue is full.")
def getMove():
    """
    Retrieve and remove the oldest move from the queue.
    """
    if not moves_queue.empty():
        move = moves_queue.get()
        
        return move
    else:
        print("No moves in the queue to retrieve.")
        return None
