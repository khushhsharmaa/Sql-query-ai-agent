import requests, time, traceback
base='http://127.0.0.1:8000'
payload={'question':'show all employee'}
print('starting POST /query')
try:
    t0=time.monotonic()
    r=requests.post(base+'/query', json=payload, timeout=120)
    t1=time.monotonic()
    print('/query status', r.status_code)
    print('elapsed={:.3f}s'.format(t1-t0))
    print('response:', r.text[:4000])
except Exception:
    t1=time.monotonic()
    print('Request failed after {:.3f}s'.format(t1-t0))
    traceback.print_exc()
