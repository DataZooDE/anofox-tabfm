# /// script
# requires-python = ">=3.10"
# dependencies = ["onnx"]
# ///
"""Regenerate docs/GGML_LOWERING.md from the lowering table in src/include/tabfm_ggml_ops.hpp and the bundled CPU graphs.

Run from the repo root:  uv run tools/ggml_parity/gen_lowering_doc.py
"""
import glob,collections,re,onnx
hdr=open("src/include/tabfm_ggml_ops.hpp").read()
rows=re.findall(r'\{"(\w+)", \{GgmlLowerKind::(\w+), "((?:[^"\\]|\\.)*)"\}\}',hdr)
nodes=collections.Counter(); graphs=collections.Counter(); n15=collections.Counter()
fs=[f for f in sorted(glob.glob("resources/graph_*.onnx")) if "graph_ext_" not in f and "graph_migraphx_" not in f]
for f in fs:
    m=onnx.load(f,load_external_data=False); c=collections.Counter(n.op_type for n in m.graph.node)
    nodes.update(c); graphs.update(set(c))
    if f.endswith("graph_nori15_regression.onnx"): n15=c
tot=sum(nodes.values()); t15=sum(n15.values())
kinds=collections.defaultdict(list)
for o,k,h in rows: kinds[k].append(o)
out=["# ggml lowering matrix (P1)\n",
"Generated from `src/include/tabfm_ggml_ops.hpp` (`GgmlLoweringTable()`) and the %d bundled CPU graphs (`resources/graph_*.onnx` without the ext and migraphx variants); Regenerate with `uv run tools/ggml_parity/gen_lowering_doc.py`. Reasons were checked against the vendored `vendor/ggml/include/ggml.h`.\n"%len(fs),
"%d distinct ops over %d graphs, %d nodes in total (Nori 1.5: %d nodes).\n"%(len(rows),len(fs),tot,t15),
"| kind | ops | nodes (all graphs) | nodes (Nori 1.5) | meaning |","|---|---|---|---|---|"]
meaning={"NATIVE":"one ggml op plus views","COMPOSITE":"a few ggml ops, or a native op with a workaround","CUSTOM":"no ggml op: host kernel inside the graph","HOST":"never reaches ggml: folded by the shape pass"}
for k in ["NATIVE","COMPOSITE","CUSTOM","HOST"]:
    ops=kinds[k]; out.append("| %s | %d | %d (%.1f%%) | %d (%.1f%%) | %s |"%(k,len(ops),sum(nodes[o] for o in ops),100*sum(nodes[o] for o in ops)/tot,sum(n15[o] for o in ops),100*sum(n15[o] for o in ops)/t15,meaning[k]))
out+=["","## Cross-cutting caveats","",
"- **Integer and bool tensors.** ggml has no bool type and its CPU kernels abort on I32/I64 for arithmetic. Integer values that derive from shapes are folded on the host (HOST); data-driven integer work (Gather/Slice/Mod on index tensors) runs as custom kernels or I32 masks.",
"- **NaN.** The graphs mask with NaN. Native ggml ops have their own NaN behaviour (clamp and min/max macros swallow it), so every NATIVE and COMPOSITE op needs a NaN-in golden in P3, not only the custom ones.",
"- **Rank.** ggml is 4-D at most; five- and six-dimensional tensors go through the rank-reduction pass (P2) and residual cases become custom ops.",
"- **Counts** below are nodes in the 45 bundled CPU graphs; they are generated, so reasons carry no numbers.","","| op | kind | graphs | nodes | Nori 1.5 | how / why |","|---|---|---|---|---|---|"]
for o,k,h in sorted(rows,key=lambda r:(["NATIVE","COMPOSITE","CUSTOM","HOST"].index(r[1]),-nodes[r[0]])):
    out.append("| %s | %s | %d/%d | %d | %d | %s |"%(o,k,graphs[o],len(fs),nodes[o],n15[o],h.replace("|","\\|")))
open("docs/GGML_LOWERING.md","w").write("\n".join(out)+"\n")
print("\n".join(out[:9]))
