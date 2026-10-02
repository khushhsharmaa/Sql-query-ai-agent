import requests, time, traceback

base='http://127.0.0.1:8000'
print('Testing', base)
# /health
try:
    t0=time.monotonic()
    r=requests.get(base+'/health', timeout=10)
    t1=time.monotonic()
    print('/health', r.status_code, r.text.strip(), 'elapsed={:.3f}s'.format(t1-t0))
except Exception:
    print('Health check failed')
    traceback.print_exc()

# POST /query
payload={'question':'show all employee'}
print('\nPOST /query payload:', payload)
try:
    t0=time.monotonic()
    r=requests.post(base+'/query', json=payload, timeout=180)
    t1=time.monotonic()
    print('/query status', r.status_code)
    print('elapsed={:.3f}s'.format(t1-t0))
    print('response:', r.text[:4000])
except Exception:
    t1=time.monotonic()
    print('Request failed after {:.3f}s'.format(t1-t0))
    traceback.print_exc()
