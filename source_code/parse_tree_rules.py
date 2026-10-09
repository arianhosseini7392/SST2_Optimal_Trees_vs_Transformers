import re
import sys


def parse_tree_to_rules(tree_text):
    """Convert pystreed print_tree() output into a list of (conditions, label, depth) rules."""
    lines = [l for l in tree_text.splitlines() if l.strip()]
    stack = []  # list of (depth, condition_str)
    rules = []

    for line in lines:
        depth = line.count("|")
        content = line.lstrip("| ").strip()

        # trim stack to current depth
        stack = [s for s in stack if s[0] < depth]

        if content.startswith("Label:"):
            label = content.split(":")[1].strip()
            conditions = [c for _, c in stack]
            rules.append({"conditions": conditions, "label": label, "depth": depth})
        else:
            m = re.match(r"(.+?) (not satisfied|satisfied)$", content)
            feature_expr, status = m.group(1), m.group(2)
            # feature_expr looks like "afinn_score <= 1.0"
            fm = re.match(r"(.+?) <= ([\-\d\.]+)", feature_expr)
            if fm:
                feat, thresh = fm.group(1), fm.group(2)
                if status == "satisfied":
                    cond = f"{feat} <= {thresh}"
                else:
                    cond = f"{feat} > {thresh}"
            else:
                cond = feature_expr if status == "satisfied" else f"NOT ({feature_expr})"
            stack.append((depth, cond))

    return rules


def rules_to_text(rules, label_map=None):
    label_map = label_map or {"0": "Negative", "1": "Positive"}
    out = []
    for r in rules:
        cond_str = " AND ".join(r["conditions"]) if r["conditions"] else "(always)"
        pred = label_map.get(r["label"], r["label"])
        out.append(f"IF {cond_str}\n  THEN {pred}")
    return "\n\n".join(out)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "../results/trees/hybrid_depth4.txt"
    with open(path) as f:
        text = f.read()
    rules = parse_tree_to_rules(text)
    print(rules_to_text(rules))
    print(f"\n\n{len(rules)} rules total")
