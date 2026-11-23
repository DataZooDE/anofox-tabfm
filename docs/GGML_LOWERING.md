# ggml lowering matrix (P1)

Generated from `vendor/anofox-ggml-adapter/include/anofox_ggml_adapter/ggml_ops.hpp` (`GgmlLoweringTable()`) and the 45 bundled CPU graphs (`resources/graph_*.onnx` without the ext variants); Regenerate with `uv run tools/ggml_parity/gen_lowering_doc.py`. Reasons were checked against the vendored `vendor/anofox-ggml-adapter/vendor/ggml/include/ggml.h`.

72 distinct ops over 45 graphs, 402814 nodes in total (Nori 1.5: 12592 nodes).

| kind | ops | nodes (all graphs) | nodes (Nori 1.5) | meaning |
|---|---|---|---|---|
| NATIVE | 23 | 170304 (42.3%) | 5534 (43.9%) | one ggml op plus views |
| COMPOSITE | 16 | 87513 (21.7%) | 2483 (19.7%) | a few ggml ops, or a native op with a workaround |
| CUSTOM | 25 | 15028 (3.7%) | 398 (3.2%) | no ggml op: host kernel inside the graph |
| HOST | 8 | 129969 (32.3%) | 4177 (33.2%) | never reaches ggml: folded by the shape pass |

## Cross-cutting caveats

- **Integer and bool tensors.** ggml has no bool type and its CPU kernels abort on I32/I64 for arithmetic. Integer values that derive from shapes are folded on the host (HOST); data-driven integer work (Gather/Slice/Mod on index tensors) runs as custom kernels or I32 masks.
- **NaN.** The graphs mask with NaN. Native ggml ops have their own NaN behaviour (clamp and min/max macros swallow it), so every NATIVE and COMPOSITE op needs a NaN-in golden in P3, not only the custom ones.
- **Rank.** ggml is 4-D at most; five- and six-dimensional tensors go through the rank-reduction pass (P2) and residual cases become custom ops.
- **Counts** below are nodes in the 45 bundled CPU graphs; they are generated, so reasons carry no numbers.

| op | kind | graphs | nodes | Nori 1.5 | how / why |
|---|---|---|---|---|---|
| Reshape | NATIVE | 45/45 | 62643 | 1966 | ggml_reshape_1d..4d; five- and six-dimensional shapes go through the rank-reduction pass first |
| Concat | NATIVE | 45/45 | 22594 | 969 | ggml_concat along the (permuted) axis |
| Transpose | NATIVE | 45/45 | 22209 | 672 | ggml_permute followed by ggml_cont where a contiguous layout is needed |
| Mul | NATIVE | 45/45 | 21154 | 683 | ggml_mul; src1 broadcasts into src0, operands swapped when needed. Float tensors only: the ggml CPU kernels abort on I32/I64, so integer arithmetic is HOST (folded) or a custom kernel. |
| Add | NATIVE | 45/45 | 18199 | 327 | ggml_add; src1 broadcasts into src0, so the lowering swaps operands when the larger one is on the right. Float tensors only: the ggml CPU kernels abort on I32/I64, so integer arithmetic is HOST (folded) or a custom kernel. |
| Sqrt | NATIVE | 45/45 | 9545 | 395 | ggml_sqrt |
| Unsqueeze | NATIVE | 45/45 | 5116 | 178 | a reshape view |
| Expand | NATIVE | 45/45 | 4601 | 81 | ggml_repeat to the broadcast shape |
| Log | NATIVE | 30/45 | 1090 | 62 | ggml_log |
| Squeeze | NATIVE | 45/45 | 806 | 160 | a reshape view |
| Tanh | NATIVE | 28/45 | 694 | 1 | GGML_UNARY_OP_TANH |
| Sigmoid | NATIVE | 14/45 | 548 | 33 | GGML_UNARY_OP_SIGMOID |
| Cos | NATIVE | 26/45 | 269 | 0 | ggml_cos |
| Sin | NATIVE | 26/45 | 269 | 0 | ggml_sin |
| Softplus | NATIVE | 4/45 | 168 | 0 | ggml_softplus |
| Exp | NATIVE | 7/45 | 161 | 1 | GGML_UNARY_OP_EXP |
| Neg | NATIVE | 26/45 | 155 | 1 | GGML_UNARY_OP_NEG on float tensors; integer negation is HOST or custom |
| Sign | NATIVE | 9/45 | 30 | 0 | ggml_sgn |
| Abs | NATIVE | 10/45 | 20 | 2 | GGML_UNARY_OP_ABS |
| CumSum | NATIVE | 10/45 | 16 | 0 | ggml_cumsum along dim 0; other axes permute first |
| Ceil | NATIVE | 7/45 | 8 | 1 | GGML_UNARY_OP_CEIL |
| Floor | NATIVE | 7/45 | 8 | 2 | GGML_UNARY_OP_FLOOR |
| Relu | NATIVE | 1/45 | 1 | 0 | GGML_UNARY_OP_RELU |
| Cast | COMPOSITE | 45/45 | 24803 | 442 | ggml_cast between F32/F16/I32; int64 and bool values live on the host or as I32 masks |
| Slice | COMPOSITE | 45/45 | 14711 | 407 | ggml_view_4d with strides, then ggml_cont; negative steps need a copy through a custom kernel |
| MatMul | COMPOSITE | 45/45 | 14174 | 411 | ggml_mul_mat on a transposed right operand; batch broadcast only where src0's batch divides src1's |
| CastLike | COMPOSITE | 45/45 | 11400 | 330 | same as Cast with the target dtype taken from the second operand |
| Div | COMPOSITE | 45/45 | 4428 | 73 | ggml_div; ggml_repeat of the left operand when the right one is the larger. Float tensors only: the ggml CPU kernels abort on I32/I64, so integer arithmetic is HOST (folded) or a custom kernel. |
| ReduceMean | COMPOSITE | 35/45 | 4316 | 214 | ggml_mean over dim 0 after permuting or merging the reduced axes |
| Reciprocal | COMPOSITE | 32/45 | 4221 | 213 | ggml_div of a constant ones tensor by x |
| Gather | COMPOSITE | 42/45 | 2312 | 244 | ggml_get_rows with I32 indices on dim 0 (int64 indices are cast first); other axes permute first |
| Softmax | COMPOSITE | 45/45 | 2020 | 60 | ggml_soft_max works on dim 0; other axes permute first |
| LayerNormalization | COMPOSITE | 29/45 | 1662 | 5 | ggml_norm, then ggml_mul by scale and ggml_add of bias |
| Gemm | COMPOSITE | 29/45 | 1210 | 0 | ggml_mul_mat + ggml_add bias, with alpha/beta/transA/transB handled by layout |
| ReduceSum | COMPOSITE | 43/45 | 927 | 70 | ggml_sum_rows over dim 0 after permuting or merging the reduced axes |
| Sub | COMPOSITE | 35/45 | 808 | 13 | ggml_sub; ggml_repeat of the left operand when the right one is the larger. Float tensors only: the ggml CPU kernels abort on I32/I64, so integer arithmetic is HOST (folded) or a custom kernel. |
| Einsum | COMPOSITE | 11/45 | 476 | 0 | permutes + ggml_mul_mat for the equations the bundled graphs use (verified in P3) |
| Pad | COMPOSITE | 24/45 | 36 | 0 | ggml_pad_ext zero-fills only: a non-zero constant_value needs an added mask, and reflect/edge modes have no ggml op (modes in use checked in P3) |
| TopK | COMPOSITE | 9/45 | 9 | 1 | ggml_argsort_top_k plus ggml_get_rows for the values: ggml_top_k itself returns indices in no particular order (its CPU kernel swaps the first two) where ONNX TopK is sorted; tie order checked in P3 |
| Pow | CUSTOM | 35/45 | 4361 | 217 | no ggml pow: constant exponents 2, 0.5 and 3 lower to sqr/sqrt/sqr*x (nearly all nodes), only the dynamic-exponent nodes need the kernel |
| Where | CUSTOM | 45/45 | 2413 | 23 | no ggml select; arithmetic emulation is wrong because masked values are NaN (NaN*0 = NaN) |
| Erf | CUSTOM | 41/45 | 2039 | 3 | only GGML_UNARY_OP_GELU_ERF exists (x*0.5*(1+erf(x/sqrt2)), singular to invert at 0) |
| Clip | CUSTOM | 43/45 | 1291 | 67 | ggml_clamp is MAX(MIN(x,hi),lo), which turns NaN into hi, while ONNX Clip passes NaN through and the graphs mask with NaN; the bounds are folded constants, the kernel keeps NaN |
| Less | CUSTOM | 32/45 | 704 | 6 | no ggml comparison ops; host kernel producing an I32 mask |
| And | CUSTOM | 32/45 | 684 | 16 | no ggml logical ops; host kernel over I32 masks (NaN-safe masks are the point of the graphs) |
| IsNaN | CUSTOM | 34/45 | 649 | 11 | no ggml predicate ops; host kernel producing an I32 mask |
| LessOrEqual | CUSTOM | 7/45 | 583 | 0 | no ggml comparison ops; host kernel producing an I32 mask |
| GatherElements | CUSTOM | 24/45 | 484 | 2 | no per-element gather along an axis in ggml; host kernel |
| Max | CUSTOM | 31/45 | 343 | 2 | tensor-tensor maximum has no ggml op; the scalar-constant form cannot use ggml_clamp either (NaN becomes the bound) |
| Greater | CUSTOM | 25/45 | 246 | 6 | no ggml comparison ops; host kernel producing an I32 mask |
| Not | CUSTOM | 34/45 | 219 | 19 | no ggml logical ops; host kernel over I32 masks |
| Min | CUSTOM | 31/45 | 202 | 2 | tensor-tensor minimum has no ggml op; the scalar-constant form cannot use ggml_clamp either (NaN becomes the bound) |
| IsInf | CUSTOM | 28/45 | 180 | 12 | no ggml predicate ops; host kernel producing an I32 mask |
| GreaterOrEqual | CUSTOM | 29/45 | 174 | 1 | no ggml comparison ops; host kernel producing an I32 mask |
| Equal | CUSTOM | 38/45 | 146 | 3 | no ggml comparison ops; host kernel producing an I32 mask |
| ScatterND | CUSTOM | 16/45 | 98 | 0 | no n-d scatter in ggml; host kernel |
| Mod | CUSTOM | 22/45 | 58 | 2 | no ggml modulo; ONNX integer Mod takes the sign of the divisor, host kernel |
| GatherND | CUSTOM | 10/45 | 54 | 0 | no n-d index gather in ggml; host kernel |
| ReduceMin | CUSTOM | 17/45 | 47 | 2 | ggml has no min reduction; host kernel |
| Or | CUSTOM | 11/45 | 25 | 2 | no ggml logical ops; host kernel over I32 masks |
| ReduceMax | CUSTOM | 6/45 | 9 | 2 | ggml has argmax but no max reduction; host kernel |
| Round | CUSTOM | 4/45 | 8 | 0 | ggml_round rounds half away from zero, ONNX rounds half to even |
| ScatterElements | CUSTOM | 7/45 | 7 | 0 | no scatter along an axis in ggml (ggml_set_rows covers rows only); host kernel |
| ReduceProd | CUSTOM | 4/45 | 4 | 0 | ggml has no product reduction; host kernel |
| Constant | HOST | 45/45 | 118964 | 4027 | folded into a constant tensor by the shape pass |
| Shape | HOST | 45/45 | 5489 | 139 | evaluated by the shape pass |
| Identity | HOST | 38/45 | 3088 | 6 | an alias, resolved while lowering |
| Range | HOST | 39/45 | 935 | 1 | folded: its bounds come from Shape values the shape pass knows |
| SequenceAt | HOST | 18/45 | 932 | 2 | sequences are resolved to views at lowering time |
| SplitToSequence | HOST | 18/45 | 412 | 1 | resolved to views at lowering time |
| ConstantOfShape | HOST | 13/45 | 137 | 1 | folded: the shape operand is known per input shape |
| Size | HOST | 2/45 | 12 | 0 | evaluated by the shape pass |
