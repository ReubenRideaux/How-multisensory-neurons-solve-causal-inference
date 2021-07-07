import sys, importlib

try:
    sys.argv[1]
except:
    print('No input argument provided.')
else:
    if sys.argv[1][-3:]=='.py':
        sys.argv[1]=sys.argv[1][:-3]
    module = importlib.import_module(sys.argv[1])
    # module.main('')
