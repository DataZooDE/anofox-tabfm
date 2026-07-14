//===----------------------------------------------------------------------===//
// tabfm_ckpt.cpp — native PyTorch .ckpt reader (zip + subset pickle VM).
//===----------------------------------------------------------------------===//

#include "tabfm_ckpt.hpp"

#include "duckdb/common/exception.hpp"

#include <cstring>

namespace duckdb {
namespace anofox {

namespace {

//===--------------------------------------------------------------------===//
// Minimal ZIP reader (STORED entries only — torch.save uses no compression).
//===--------------------------------------------------------------------===//

uint16_t RdU16(const uint8_t *p) {
	return (uint16_t)(p[0] | (p[1] << 8));
}
uint32_t RdU32(const uint8_t *p) {
	return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
uint64_t RdU64(const uint8_t *p) {
	return (uint64_t)RdU32(p) | ((uint64_t)RdU32(p + 4) << 32);
}

struct ZipEntry {
	const uint8_t *data;
	idx_t size;
};

// filename (zip-internal) -> raw stored bytes.
unordered_map<string, ZipEntry> ReadZip(const uint8_t *buf, idx_t size) {
	if (size < 22) {
		throw InvalidInputException("tabfm: checkpoint is not a valid zip archive (too small)");
	}
	// End-of-central-directory: scan back for signature 0x06054b50.
	idx_t eocd = 0;
	bool found = false;
	idx_t scan_from = size - 22;
	idx_t scan_to = size > (22 + 65536) ? size - 22 - 65536 : 0;
	for (idx_t i = scan_from + 1; i-- > scan_to;) {
		if (RdU32(buf + i) == 0x06054b50) {
			eocd = i;
			found = true;
			break;
		}
	}
	if (!found) {
		throw InvalidInputException("tabfm: checkpoint zip end-of-central-directory not found");
	}
	uint32_t cd_count = RdU16(buf + eocd + 10);
	uint32_t cd_offset = RdU32(buf + eocd + 16);
	// ZIP64 sentinel: torch checkpoints of the sizes we ship stay under 4 GB.
	if (cd_offset == 0xFFFFFFFFu || cd_count == 0xFFFFu) {
		throw InvalidInputException("tabfm: ZIP64 checkpoints are not supported");
	}

	unordered_map<string, ZipEntry> entries;
	idx_t p = cd_offset;
	for (uint32_t i = 0; i < cd_count; i++) {
		if (p + 46 > size || RdU32(buf + p) != 0x02014b50) {
			throw InvalidInputException("tabfm: malformed checkpoint zip central directory");
		}
		uint16_t method = RdU16(buf + p + 10);
		uint32_t comp_size = RdU32(buf + p + 20);
		uint16_t fn_len = RdU16(buf + p + 28);
		uint16_t extra_len = RdU16(buf + p + 30);
		uint16_t cmt_len = RdU16(buf + p + 32);
		uint32_t lho = RdU32(buf + p + 42); // local header offset
		string name((const char *)(buf + p + 46), fn_len);
		if (method != 0) {
			throw InvalidInputException("tabfm: checkpoint zip entry '%s' is compressed (only STORED supported)",
			                            name);
		}
		// Local header: 30 bytes + its own filename + extra, then the data.
		if (lho + 30 > size || RdU32(buf + lho) != 0x04034b50) {
			throw InvalidInputException("tabfm: malformed checkpoint zip local header for '%s'", name);
		}
		uint16_t l_fn = RdU16(buf + lho + 26);
		uint16_t l_extra = RdU16(buf + lho + 28);
		idx_t data_off = lho + 30 + l_fn + l_extra;
		if (data_off + comp_size > size) {
			throw InvalidInputException("tabfm: checkpoint zip entry '%s' runs past end of file", name);
		}
		entries[name] = ZipEntry {buf + data_off, comp_size};
		p += 46 + fn_len + extra_len + cmt_len;
	}
	return entries;
}

//===--------------------------------------------------------------------===//
// Subset pickle VM (protocol 2..4, the ops torch.save emits for a state_dict).
//===--------------------------------------------------------------------===//

struct PVal {
	enum K { EMPTY, MARK, NONE, INT, FLOAT, STR, TUPLE, DICT, GLOBAL, PERSID, TENSOR } k = EMPTY;
	int64_t i = 0;
	string s;              // STR text; GLOBAL "module.name"; PERSID storage-type name
	string skey;           // PERSID / TENSOR: storage key (the data/<skey> blob)
	vector<PVal> items;    // TUPLE items; DICT: flat [k0,v0,k1,v1,...]
	vector<int64_t> shape; // TENSOR shape
};

string StorageDtype(const string &storage_type) {
	// storage_type is e.g. "torch.FloatStorage".
	auto dot = storage_type.rfind('.');
	string name = dot == string::npos ? storage_type : storage_type.substr(dot + 1);
	if (name == "FloatStorage") return "f32";
	if (name == "HalfStorage") return "f16";
	if (name == "BFloat16Storage") return "bf16";
	if (name == "DoubleStorage") return "f64";
	if (name == "LongStorage") return "i64";
	if (name == "IntStorage") return "i32";
	if (name == "ShortStorage") return "i16";
	if (name == "CharStorage") return "i8";
	if (name == "ByteStorage") return "u8";
	if (name == "BoolStorage") return "bool";
	throw InvalidInputException("tabfm: unsupported checkpoint storage type '%s'", storage_type);
}

class Unpickler {
public:
	Unpickler(const uint8_t *data, idx_t size) : p_(data), end_(data + size) {
	}

	PVal Run() {
		while (p_ < end_) {
			uint8_t op = *p_++;
			switch (op) {
			case 0x80: // PROTO
				Need(1);
				p_ += 1;
				break;
			case 0x95: // FRAME
				Need(8);
				p_ += 8;
				break;
			case '}': // EMPTY_DICT
				Push(PVal {PVal::DICT});
				break;
			case ']': // EMPTY_LIST -> treat as empty tuple (we never index lists)
			case ')': // EMPTY_TUPLE
				Push(PVal {PVal::TUPLE});
				break;
			case '(': // MARK
				Push(PVal {PVal::MARK});
				break;
			case 'N': // NONE
				Push(PVal {PVal::NONE});
				break;
			case 0x88: // NEWTRUE
				PushInt(1);
				break;
			case 0x89: // NEWFALSE
				PushInt(0);
				break;
			case 'K': // BININT1
				Need(1);
				PushInt(p_[0]);
				p_ += 1;
				break;
			case 'M': // BININT2
				Need(2);
				PushInt(RdU16(p_));
				p_ += 2;
				break;
			case 'J': // BININT (signed 4)
				Need(4);
				PushInt((int32_t)RdU32(p_));
				p_ += 4;
				break;
			case 0x8a: { // LONG1
				Need(1);
				uint8_t n = p_[0];
				p_ += 1;
				Need(n);
				int64_t v = 0;
				for (int b = 0; b < n; b++) {
					v |= (int64_t)p_[b] << (8 * b);
				}
				if (n > 0 && n < 8 && (p_[n - 1] & 0x80)) { // sign-extend
					v |= -(int64_t)1 << (8 * n);
				}
				p_ += n;
				PushInt(v);
				break;
			}
			case 'G': { // BINFLOAT (big-endian double)
				Need(8);
				uint8_t tmp[8];
				for (int b = 0; b < 8; b++) {
					tmp[b] = p_[7 - b];
				}
				double d;
				std::memcpy(&d, tmp, 8);
				PVal v {PVal::FLOAT};
				v.i = 0;
				Push(std::move(v)); // value unused
				p_ += 8;
				break;
			}
			case 'X': { // BINUNICODE (4-byte len)
				Need(4);
				uint32_t n = RdU32(p_);
				p_ += 4;
				Need(n);
				PVal v {PVal::STR};
				v.s.assign((const char *)p_, n);
				p_ += n;
				Push(std::move(v));
				break;
			}
			case 0x8c: { // SHORT_BINUNICODE (1-byte len)
				Need(1);
				uint8_t n = p_[0];
				p_ += 1;
				Need(n);
				PVal v {PVal::STR};
				v.s.assign((const char *)p_, n);
				p_ += n;
				Push(std::move(v));
				break;
			}
			case 'q': // BINPUT
				Need(1);
				Memo(p_[0]);
				p_ += 1;
				break;
			case 'r': // LONG_BINPUT
				Need(4);
				Memo(RdU32(p_));
				p_ += 4;
				break;
			case 0x94: // MEMOIZE
				Memo(memo_.size());
				break;
			case 'h': // BINGET
				Need(1);
				Push(GetMemo(p_[0]));
				p_ += 1;
				break;
			case 'j': // LONG_BINGET
				Need(4);
				Push(GetMemo(RdU32(p_)));
				p_ += 4;
				break;
			case 'c': { // GLOBAL (module\nname\n)
				string mod = ReadLine();
				string name = ReadLine();
				PVal v {PVal::GLOBAL};
				v.s = mod + "." + name;
				Push(std::move(v));
				break;
			}
			case 0x93: { // STACK_GLOBAL
				PVal name = Pop();
				PVal mod = Pop();
				PVal v {PVal::GLOBAL};
				v.s = mod.s + "." + name.s;
				Push(std::move(v));
				break;
			}
			case 0x85: // TUPLE1
				MakeTuple(1);
				break;
			case 0x86: // TUPLE2
				MakeTuple(2);
				break;
			case 0x87: // TUPLE3
				MakeTuple(3);
				break;
			case 't': // TUPLE (to MARK)
				MakeTupleToMark();
				break;
			case 's': // SETITEM
				SetItem();
				break;
			case 'u': // SETITEMS (to MARK)
				SetItems();
				break;
			case 'b': { // BUILD (pop state; keep obj)
				Pop();
				break;
			}
			case 'Q': // BINPERSID
				MakePersid();
				break;
			case 'R': // REDUCE
				Reduce();
				break;
			case '.': // STOP
				return Pop();
			default:
				throw InvalidInputException("tabfm: unsupported pickle opcode 0x%02x in checkpoint", (int)op);
			}
		}
		throw InvalidInputException("tabfm: checkpoint pickle ended without STOP");
	}

private:
	const uint8_t *p_;
	const uint8_t *end_;
	vector<PVal> stack_;
	vector<PVal> memo_;

	void Need(idx_t n) {
		if (p_ + n > end_) {
			throw InvalidInputException("tabfm: truncated checkpoint pickle");
		}
	}
	void Push(PVal v) {
		stack_.push_back(std::move(v));
	}
	void PushInt(int64_t v) {
		PVal x {PVal::INT};
		x.i = v;
		Push(std::move(x));
	}
	PVal Pop() {
		if (stack_.empty()) {
			throw InvalidInputException("tabfm: checkpoint pickle stack underflow");
		}
		PVal v = std::move(stack_.back());
		stack_.pop_back();
		return v;
	}
	void Memo(idx_t idx) {
		if (stack_.empty()) {
			throw InvalidInputException("tabfm: checkpoint pickle memo of empty stack");
		}
		if (idx >= memo_.size()) {
			memo_.resize(idx + 1);
		}
		memo_[idx] = stack_.back();
	}
	PVal GetMemo(idx_t idx) {
		if (idx >= memo_.size()) {
			throw InvalidInputException("tabfm: checkpoint pickle memo index out of range");
		}
		return memo_[idx];
	}
	string ReadLine() {
		const uint8_t *start = p_;
		while (p_ < end_ && *p_ != '\n') {
			p_++;
		}
		string s((const char *)start, p_ - start);
		if (p_ < end_) {
			p_++; // skip newline
		}
		return s;
	}
	void MakeTuple(idx_t n) {
		PVal t {PVal::TUPLE};
		t.items.resize(n);
		for (idx_t i = 0; i < n; i++) {
			t.items[n - 1 - i] = Pop();
		}
		Push(std::move(t));
	}
	idx_t MarkPos() {
		for (idx_t i = stack_.size(); i-- > 0;) {
			if (stack_[i].k == PVal::MARK) {
				return i;
			}
		}
		throw InvalidInputException("tabfm: checkpoint pickle MARK not found");
	}
	void MakeTupleToMark() {
		idx_t m = MarkPos();
		PVal t {PVal::TUPLE};
		for (idx_t i = m + 1; i < stack_.size(); i++) {
			t.items.push_back(std::move(stack_[i]));
		}
		stack_.resize(m); // drop mark + items
		Push(std::move(t));
	}
	void SetItem() {
		PVal v = Pop();
		PVal k = Pop();
		if (stack_.empty() || stack_.back().k != PVal::DICT) {
			throw InvalidInputException("tabfm: checkpoint pickle SETITEM without dict");
		}
		stack_.back().items.push_back(std::move(k));
		stack_.back().items.push_back(std::move(v));
	}
	void SetItems() {
		idx_t m = MarkPos();
		vector<PVal> pairs;
		for (idx_t i = m + 1; i < stack_.size(); i++) {
			pairs.push_back(std::move(stack_[i]));
		}
		stack_.resize(m); // drop mark + items
		if (stack_.empty() || stack_.back().k != PVal::DICT) {
			throw InvalidInputException("tabfm: checkpoint pickle SETITEMS without dict");
		}
		for (auto &x : pairs) {
			stack_.back().items.push_back(std::move(x));
		}
	}
	void MakePersid() {
		// pid tuple: ('storage', <GLOBAL StorageType>, key, location, numel)
		PVal pid = Pop();
		if (pid.k != PVal::TUPLE || pid.items.size() < 5) {
			throw InvalidInputException("tabfm: unexpected checkpoint persistent id");
		}
		PVal out {PVal::PERSID};
		out.s = pid.items[1].s;    // storage type ("torch.FloatStorage")
		out.skey = pid.items[2].s; // storage key
		out.i = pid.items[4].i;    // numel
		Push(std::move(out));
	}
	void Reduce() {
		PVal args = Pop();
		PVal fn = Pop();
		if (fn.k != PVal::GLOBAL) {
			// e.g. OrderedDict()/set() — produce an empty dict placeholder.
			Push(PVal {PVal::DICT});
			return;
		}
		if (fn.s == "torch._utils._rebuild_tensor_v2" || fn.s == "torch._utils._rebuild_tensor") {
			// args: (storage PERSID, storage_offset, size TUPLE, stride TUPLE, ...)
			if (args.items.size() < 3 || args.items[0].k != PVal::PERSID) {
				throw InvalidInputException("tabfm: unexpected _rebuild_tensor args in checkpoint");
			}
			const PVal &storage = args.items[0];
			int64_t offset = args.items.size() > 1 ? args.items[1].i : 0;
			PVal t {PVal::TENSOR};
			t.s = storage.s;       // storage type
			t.skey = storage.skey; // storage key
			t.i = offset;
			for (auto &d : args.items[2].items) {
				t.shape.push_back(d.i);
			}
			Push(std::move(t));
			return;
		}
		if (fn.s == "torch._utils._rebuild_parameter") {
			// args: (data TENSOR, requires_grad, hooks) → the data tensor.
			if (args.items.empty()) {
				throw InvalidInputException("tabfm: unexpected _rebuild_parameter args");
			}
			Push(std::move(args.items[0]));
			return;
		}
		if (fn.s == "collections.OrderedDict") {
			Push(PVal {PVal::DICT});
			return;
		}
		// Unknown callable: a placeholder none (we only read tensors + dicts).
		Push(PVal {PVal::NONE});
	}
};

idx_t DtypeSize(const string &dt) {
	if (dt == "f64" || dt == "i64") return 8;
	if (dt == "f32" || dt == "i32") return 4;
	if (dt == "f16" || dt == "bf16" || dt == "i16") return 2;
	return 1; // i8 / u8 / bool
}

} // namespace

bool IsTorchCkpt(const uint8_t *buf, idx_t size) {
	return size >= 4 && buf[0] == 'P' && buf[1] == 'K' && buf[2] == 0x03 && buf[3] == 0x04;
}

unordered_map<string, CkptTensor> ReadTorchCkpt(const uint8_t *buf, idx_t size) {
	auto entries = ReadZip(buf, size);
	// Locate <root>/data.pkl and the <root>/data/ prefix.
	const ZipEntry *pkl = nullptr;
	string root;
	for (auto &e : entries) {
		if (e.first.size() >= 8 && e.first.compare(e.first.size() - 8, 8, "data.pkl") == 0) {
			pkl = &e.second;
			root = e.first.substr(0, e.first.size() - 8); // "<root>/"
			break;
		}
	}
	if (!pkl) {
		throw InvalidInputException("tabfm: checkpoint has no data.pkl (not a torch.save archive)");
	}
	string data_prefix = root + "data/";

	Unpickler up(pkl->data, pkl->size);
	PVal top = up.Run();
	if (top.k != PVal::DICT) {
		throw InvalidInputException("tabfm: checkpoint top-level object is not a state_dict");
	}
	// Many checkpoints wrap the tensors under a "state_dict" (or "model") key,
	// alongside a config; descend into it if present.
	const PVal *sd = &top;
	for (idx_t i = 0; i + 1 < top.items.size(); i += 2) {
		if (top.items[i].k == PVal::STR && top.items[i + 1].k == PVal::DICT) {
			const string &k = top.items[i].s;
			if (k == "state_dict" || k == "model" || k == "model_state_dict" || k == "weights") {
				sd = &top.items[i + 1];
				break;
			}
		}
	}

	unordered_map<string, CkptTensor> out;
	for (idx_t i = 0; i + 1 < sd->items.size(); i += 2) {
		const PVal &key = sd->items[i];
		const PVal &val = sd->items[i + 1];
		if (key.k != PVal::STR || val.k != PVal::TENSOR) {
			continue; // skip non-tensor entries (e.g. metadata)
		}
		auto blob = entries.find(data_prefix + val.skey);
		if (blob == entries.end()) {
			throw InvalidInputException("tabfm: checkpoint storage blob '%s' missing", val.skey);
		}
		CkptTensor t;
		t.dtype = StorageDtype(val.s);
		t.shape = val.shape;
		idx_t elt = DtypeSize(t.dtype);
		idx_t numel = 1;
		for (auto d : t.shape) {
			numel *= (idx_t)(d < 0 ? 0 : d);
		}
		idx_t byte_off = (idx_t)val.i * elt;
		t.nbytes = numel * elt;
		if (byte_off + t.nbytes > blob->second.size) {
			throw InvalidInputException("tabfm: checkpoint tensor '%s' exceeds its storage blob", key.s);
		}
		t.data = blob->second.data + byte_off;
		out.emplace(key.s, std::move(t));
	}
	return out;
}

} // namespace anofox
} // namespace duckdb
