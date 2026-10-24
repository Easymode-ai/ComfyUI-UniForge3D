from typing import *

BACKEND = 'flash_attn' 
DEBUG = False

def __from_env():
    import os
    
    global BACKEND
    global DEBUG
    
    env_attn_backend = os.environ.get('ATTN_BACKEND')
    env_attn_debug = os.environ.get('ATTN_DEBUG')
    
    if env_attn_backend is not None and env_attn_backend in ['xformers', 'flash_attn', 'flash_attn_3', 'sdpa', 'naive']:
        BACKEND = env_attn_backend
    if env_attn_debug is not None:
        DEBUG = env_attn_debug == '1'

    print(f"[Attention Config] BACKEND={BACKEND}, DEBUG={DEBUG}")

__from_env()
    

def set_backend(backend: Literal['xformers', 'flash_attn']):
    global BACKEND
    BACKEND = backend
    print(f"[Attention Config] BACKEND set to {BACKEND}")

def set_debug(debug: bool):
    global DEBUG
    DEBUG = debug
    print(f"[Attention Config] DEBUG set to {DEBUG}")
