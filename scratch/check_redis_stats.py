import redis
try:
    r = redis.Redis(host='127.0.0.1', port=6379, db=0)
    info = r.info('memory')
    print(f"Used Memory: {info['used_memory_human']}")
    print(f"Max Memory: {info.get('maxmemory_human', 'Unlimited')}")
    
    clients = r.info('clients')
    print(f"Connected Clients: {clients['connected_clients']}")
except Exception as e:
    print(f"Error: {e}")
