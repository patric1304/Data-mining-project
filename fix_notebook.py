import json

with open("d:\\faculta\\DM\\Data-mining-project\\airbnb_analysis.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell["cell_type"] == "code":
        source = cell["source"]
        # Find the price cleaning cell
        if any("listings['price'].str.replace" in line for line in source):
            # Replace the condition to be more robust
            new_source = []
            for line in source:
                if "if listings['price'].dtype == 'object':" in line:
                    new_source.append("if listings['price'].dtype in ['object', 'string', 'O']:\n")
                else:
                    new_source.append(line)
            cell["source"] = new_source

with open("d:\\faculta\\DM\\Data-mining-project\\airbnb_analysis.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Notebook modified successfully.")
