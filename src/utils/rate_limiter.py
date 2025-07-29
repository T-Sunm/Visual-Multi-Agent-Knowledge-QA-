import time
import threading
from functools import wraps

# Rate limits according to API docs
ARXIV_DELAY = 3.0 
WIKIPEDIA_DELAY = 2.0 

class RateLimiter:
    def __init__(self):
        self.last_call = {}
        self.lock = threading.Lock()
    
    def rate_limit(self, service: str, delay: float):
        """Rate limiting decorator"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                with self.lock:
                    current_time = time.time()
                    if service in self.last_call:
                        time_since_last = current_time - self.last_call[service]
                        if time_since_last < delay:
                            sleep_time = delay - time_since_last
                            print(f"Rate limiting {service}: sleeping {sleep_time:.2f}s")
                            time.sleep(sleep_time)
                    
                    self.last_call[service] = time.time()
                    return func(*args, **kwargs)
            return wrapper
        return decorator

# Global rate limiter instance
rate_limiter = RateLimiter()
