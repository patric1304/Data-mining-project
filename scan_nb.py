import json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
with open('d:/faculta/DM/Data-mining-project/airbnb_analysis.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)
cells = nb['cells']
for i in [136, 139, 144, 146]:
    print(f'=== Cell {i} ({cells[i]["cell_type"]}) ===')
    print(''.join(cells[i]['source']))
    print()
