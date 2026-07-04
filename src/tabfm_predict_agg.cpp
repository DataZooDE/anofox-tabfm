#include "tabfm_registration.hpp"
#include "anofox_function_alias.hpp"

#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/parser/parsed_data/create_aggregate_function_info.hpp"

namespace duckdb {
namespace anofox {

// Predict core: anofox_tabfm_predict_agg(row STRUCT, target VARCHAR [, opts MAP])
// → LIST(STRUCT(cols, yhat, yhat_score, is_training[, proba])). Also carries the
// window callback for tabfm_predict_win (spec S04 / HLD §4.6). Owned by WS-E.
//
// Phase-0 state: registered with its final argument shapes but throws
// NotImplementedException at bind time.

namespace {

unique_ptr<FunctionData> StubBind(ClientContext &, AggregateFunction &, vector<unique_ptr<Expression>> &) {
	throw NotImplementedException("anofox_tabfm_predict_agg is not implemented yet");
}

AggregateFunction StubOverload(vector<LogicalType> arguments) {
	AggregateFunction func("anofox_tabfm_predict_agg", std::move(arguments), LogicalType::ANY, nullptr, nullptr,
	                       nullptr, nullptr, nullptr, nullptr, StubBind);
	return func;
}

} // anonymous namespace

void RegisterPredictAggFunction(ExtensionLoader &loader) {
	AggregateFunctionSet set("anofox_tabfm_predict_agg");
	set.AddFunction(StubOverload({LogicalType::ANY, LogicalType::VARCHAR}));
	set.AddFunction(StubOverload({LogicalType::ANY, LogicalType::VARCHAR, LogicalType::ANY}));

	CreateAggregateFunctionInfo primary_info(set);
	loader.RegisterFunction(primary_info);

	AggregateFunctionSet alias_set("tabfm_predict_agg");
	for (auto &func : set.functions) {
		auto alias_func = func;
		alias_func.name = "tabfm_predict_agg";
		alias_set.AddFunction(std::move(alias_func));
	}
	CreateAggregateFunctionInfo alias_info(std::move(alias_set));
	alias_info.alias_of = "anofox_tabfm_predict_agg";
	loader.RegisterFunction(alias_info);
}

} // namespace anofox
} // namespace duckdb
