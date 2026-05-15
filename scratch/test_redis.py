import redis
try:
    r = redis.Redis(host='localhost', port=6379, db=0)
    print(f"Ping: {r.ping()}")
    r.set('test_key', 'test_value')
    print(f"Set: {r.get('test_key')}")
except Exception as e:
    print(f"Error: {e}")
