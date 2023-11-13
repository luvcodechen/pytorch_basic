import json


def convert_ipynb_to_py(ipynb_file, py_file):
    with open(ipynb_file, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    with open(py_file, 'w', encoding='utf-8') as f:
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                f.write(''.join(cell['source']) + '\n\n')


# convert_ipynb_to_py('Stack_vs_Caocat.ipynb', 'test.py')

# convert_ipynb_to_py('cnntest.ipynb', 'test.py')
# convert_ipynb_to_py('cnntest.ipynb', 'qww.py')
# convert_ipynb_to_py('runwithGPU.ipynb', 'dsa.py')
