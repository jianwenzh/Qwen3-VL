
local_rank = None

def set_local_rank(rank):
    global local_rank
    local_rank = rank
    
def is_rank0():
    # HF uses local_rank == -1 for non-DDP
    return local_rank in (-1, 0)

def rank0_print(*args, **kwargs):
    if is_rank0():
        print(*args, **kwargs)
