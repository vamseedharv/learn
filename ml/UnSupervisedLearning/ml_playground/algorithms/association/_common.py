"""Frequent-itemset miners and association-rule presentation."""
from collections import Counter
from itertools import combinations
from math import ceil, sqrt
from pathlib import Path
import pandas as pd
import streamlit as st


def apriori(transactions, minimum, max_length):
    counts = Counter(item for basket in transactions for item in basket)
    level = {frozenset([item]) for item, count in counts.items() if count >= minimum}
    found = {s: counts[next(iter(s))] for s in level}
    for length in range(2, max_length + 1):
        candidates = {a | b for a in level for b in level if len(a | b) == length}
        candidates = {s for s in candidates if all(frozenset(t) in level for t in combinations(sorted(s), length-1))}
        counted = {s: sum(s <= basket for basket in transactions) for s in candidates}
        level = {s for s, count in counted.items() if count >= minimum}
        found.update({s: counted[s] for s in level})
        if not level:
            break
    return found


def eclat(transactions, minimum, max_length):
    vertical = {}
    for tid, basket in enumerate(transactions):
        for item in basket:
            vertical.setdefault(item, set()).add(tid)
    found = {}
    def visit(prefix, suffix):
        for index, (item, tids) in enumerate(suffix):
            current = prefix | {item}
            found[frozenset(current)] = len(tids)
            if len(current) < max_length:
                following = [(other, tids & ids) for other, ids in suffix[index+1:] if len(tids & ids) >= minimum]
                visit(current, following)
    visit(set(), sorted((item, ids) for item, ids in vertical.items() if len(ids) >= minimum))
    return found


class Node:
    def __init__(self, item=None, parent=None):
        self.item, self.parent, self.count, self.children = item, parent, 0, {}


def fp_growth(transactions, minimum, max_length):
    found = {}
    def mine(paths, prefix):
        counts = Counter()
        for path, weight in paths:
            for item in path:
                counts[item] += weight
        counts = {i: c for i, c in counts.items() if c >= minimum}
        root, header = Node(), {i: [] for i in counts}
        for path, weight in paths:
            ordered = sorted((i for i in path if i in counts), key=lambda i: (-counts[i], i))
            node = root
            for item in ordered:
                if item not in node.children:
                    node.children[item] = Node(item, node)
                    header[item].append(node.children[item])
                node = node.children[item]
                node.count += weight
        for item in sorted(counts, key=lambda i: (counts[i], i)):
            current = prefix | {item}
            found[frozenset(current)] = counts[item]
            if len(current) >= max_length:
                continue
            conditional = []
            for node in header[item]:
                ancestors, parent = [], node.parent
                while parent.item is not None:
                    ancestors.append(parent.item)
                    parent = parent.parent
                if ancestors:
                    conditional.append((ancestors, node.count))
            if conditional:
                mine(conditional, current)
    mine([(sorted(basket), 1) for basket in transactions], set())
    return found


def rules_from_itemsets(found, total, confidence):
    rows = []
    for itemset, count in found.items():
        if len(itemset) < 2:
            continue
        for length in range(1, len(itemset)):
            for subset in combinations(sorted(itemset), length):
                left = frozenset(subset)
                right = itemset - left
                conf = count / found[left]
                if conf >= confidence:
                    support = count / total
                    left_support, right_support = found[left] / total, found[right] / total
                    lift = conf / right_support
                    # When both confidence and consequent support are 1,
                    # conviction is 0/0 rather than evidence of a useful rule.
                    conviction = (
                        (1 - right_support) / (1 - conf) if conf < 1
                        else float('inf') if right_support < 1 else float('nan')
                    )
                    rows.append({
                        "Antecedent": ', '.join(sorted(left)), "Consequent": ', '.join(sorted(right)),
                        "Count": count, "Support": support, "Confidence": conf, "Lift": lift,
                        "Leverage": support - left_support * right_support,
                        "Conviction": conviction,
                        "Jaccard": support / (left_support + right_support - support),
                        "Cosine": support / sqrt(left_support * right_support),
                    })
    columns = ["Antecedent", "Consequent", "Count", "Support", "Confidence", "Lift", "Leverage", "Conviction", "Jaccard", "Cosine"]
    return pd.DataFrame(rows, columns=columns).sort_values(["Lift", "Confidence"], ascending=False)


def explanation(name, approach):
    st.write(f"## {name}")
    st.write("Association mining finds items that co-occur in transactions without a target label. A transaction is a set: repeated occurrences of an item within one basket count once.")
    st.write("### How it works")
    st.write(approach)
    st.write("Find frequent itemsets above minimum support, then split each itemset into an antecedent and consequent to generate rules passing minimum confidence.")
    st.write("### Metrics and parameter selection")
    st.write("Support(A) is the fraction of baskets containing A. Confidence(A → B) = support(A ∪ B) / support(A). Lift = confidence / support(B); above 1 indicates positive association, 1 independence, and below 1 negative association. Leverage is support(A ∪ B) - support(A) × support(B).")
    st.write("Choose support according to transaction volume and minimum useful counts. Higher thresholds reduce results and computation but can hide rare patterns. Confidence favors popular consequents, so examine lift and support together. Maximum itemset length controls complexity.")
    st.write("Conviction = (1 - support(B)) / (1 - confidence): 1 is the independence baseline, larger values indicate fewer counterexamples than expected. Perfect confidence gives infinity unless B is universal, when it is undefined. Jaccard = support(A union B) / (support(A) + support(B) - support(A union B)); cosine = support(A union B) / sqrt(support(A) * support(B)). Both range from 0 to 1 and measure overlap, without an independence baseline. Count is the number of baskets supporting both sides.")
    st.write("### Advantages")
    st.write("Produces interpretable combinations and directional rules for basket analysis, recommendations, and exploratory pattern discovery.")
    st.write("### Disadvantages")
    st.write("Low support on many items can produce exponentially many patterns. Correlation does not imply causation; popular products, repeated customers, seasonality, and multiple comparisons can mislead. Confidence is directional but is not evidence of a causal effect.")
    st.write("### Validation and interpretation")
    st.write("Validate chosen rules on independent later transactions, checking support, confidence, lift, counts, and business usefulness. Do not select and validate rules on the same data and claim generalization. This page uses a small reproducible synthetic grocery dataset for learning, not real purchasing evidence.")


def run_miner(section, algorithm, miner):
    st.subheader("1. Inspect transactions")
    path = Path(__file__).resolve().parents[2] / 'datasets' / section / f"{section}_{algorithm.lower().replace(' ', '_')}.csv"
    df = pd.read_csv(path)
    st.dataframe(df, use_container_width=True)
    transactions = [set(str(value).strip() for value in row if pd.notna(value) and str(value).strip()) for row in df.drop(columns=['Transaction_ID'], errors='ignore').itertuples(index=False, name=None)]
    if not transactions or not any(transactions):
        st.error("No items are available.")
        return
    st.write(f"{len(transactions)} baskets, {len(set().union(*transactions))} distinct items. Empty baskets remain in the support denominator.")
    st.subheader("2. Mine frequent itemsets")
    support = st.slider('Minimum support', 0.05, 1.0, 0.2, 0.05, key=f'{algorithm}_support')
    confidence = st.slider('Minimum confidence', 0.05, 1.0, 0.6, 0.05, key=f'{algorithm}_confidence')
    length = int(st.number_input('Maximum itemset length', min_value=1, max_value=5, value=3, step=1, key=f'{algorithm}_length'))
    minimum = ceil(support * len(transactions) - 1e-12)
    found = miner(transactions, minimum, length)
    rows = [{'Items': ', '.join(sorted(items)), 'Length': len(items), 'Count': count, 'Support': count/len(transactions)} for items, count in found.items()]
    if not rows:
        st.info('No frequent itemsets meet these thresholds. Try lower support.')
        return
    itemsets = pd.DataFrame(rows).sort_values(['Length', 'Support'], ascending=[True, False])
    st.write(f'Minimum transaction count: {minimum}. Itemsets found: {len(found)}.')
    st.dataframe(itemsets, hide_index=True, use_container_width=True)
    st.subheader('3. Evaluate association rules')
    rules = rules_from_itemsets(found, len(transactions), confidence)
    if rules.empty:
        st.info('No rules meet the thresholds. Try lower confidence or allow longer itemsets.')
    else:
        ranking = st.selectbox('Rank rules by', ['Lift', 'Confidence', 'Leverage', 'Conviction', 'Jaccard', 'Cosine', 'Support', 'Count'], key=f'{algorithm}_ranking')
        rules = rules.sort_values([ranking, 'Confidence'] if ranking != 'Confidence' else ['Confidence'], ascending=False, na_position='last')
        st.dataframe(rules, hide_index=True, use_container_width=True)
        st.bar_chart(rules.head(10).assign(Rule=lambda r: r['Antecedent']+' → '+r['Consequent']).set_index('Rule')['Lift'])
        st.caption(f'Top ten rules by {ranking}, showing their lift. Undefined conviction is blank; infinity means perfect confidence with a non-universal consequent. These are exploratory patterns in synthetic training transactions.')
