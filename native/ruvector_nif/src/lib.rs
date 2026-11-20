use parking_lot::RwLock;
use rustler::{Encoder, Env, NifResult, ResourceArc, Term};
use ruvector_core::distance::distance;
use ruvector_core::index::flat::FlatIndex;
use ruvector_core::index::hnsw::HnswIndex;
use ruvector_core::index::VectorIndex;
use ruvector_core::types::{
    DbOptions, DistanceMetric, HnswConfig, QuantizationConfig, SearchQuery, SearchResult,
    VectorEntry,
};
use ruvector_core::VectorDB;
use std::collections::{HashMap, HashSet};
use std::sync::{mpsc, OnceLock};

const MAX_JSON_DEPTH: usize = 128;

enum MemoryIndex {
    Flat(FlatIndex),
    Hnsw(HnswIndex),
}

impl MemoryIndex {
    fn add(&mut self, id: String, vector: Vec<f32>) -> Result<(), String> {
        match self {
            MemoryIndex::Flat(idx) => idx.add(id, vector).map_err(|e| e.to_string()),
            MemoryIndex::Hnsw(idx) => idx.add(id, vector).map_err(|e| e.to_string()),
        }
    }

    fn add_batch(&mut self, entries: Vec<(String, Vec<f32>)>) -> Result<(), String> {
        match self {
            MemoryIndex::Flat(idx) => idx.add_batch(entries).map_err(|e| e.to_string()),
            MemoryIndex::Hnsw(idx) => idx.add_batch(entries).map_err(|e| e.to_string()),
        }
    }

    fn remove(&mut self, id: &String) -> Result<bool, String> {
        match self {
            MemoryIndex::Flat(idx) => idx.remove(id).map_err(|e| e.to_string()),
            MemoryIndex::Hnsw(idx) => idx.remove(id).map_err(|e| e.to_string()),
        }
    }

    fn search(
        &self,
        query: &[f32],
        k: usize,
        ef_search: Option<usize>,
    ) -> Result<Vec<SearchResult>, String> {
        match self {
            MemoryIndex::Flat(idx) => idx.search(query, k).map_err(|e| e.to_string()),
            MemoryIndex::Hnsw(idx) => match ef_search {
                Some(ef) => idx.search_with_ef(query, k, ef).map_err(|e| e.to_string()),
                None => idx.search(query, k).map_err(|e| e.to_string()),
            },
        }
    }
}

struct InMemoryState {
    entries: HashMap<String, VectorEntry>,
    index: MemoryIndex,
}

struct InMemoryDB {
    state: RwLock<InMemoryState>,
    options: DbOptions,
}

impl InMemoryDB {
    fn new(options: DbOptions) -> Result<Self, String> {
        let index = match &options.hnsw_config {
            Some(cfg) => MemoryIndex::Hnsw(
                HnswIndex::new(options.dimensions, options.distance_metric, cfg.clone())
                    .map_err(|e| e.to_string())?,
            ),
            None => MemoryIndex::Flat(FlatIndex::new(options.dimensions, options.distance_metric)),
        };

        Ok(Self {
            state: RwLock::new(InMemoryState {
                entries: HashMap::new(),
                index,
            }),
            options,
        })
    }

    fn insert(&self, mut entry: VectorEntry) -> Result<String, String> {
        if entry.vector.len() != self.options.dimensions {
            return Err(format!(
                "Dimension mismatch: expected {}, actual {}",
                self.options.dimensions,
                entry.vector.len()
            ));
        }

        let id = entry
            .id
            .take()
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        entry.id = Some(id.clone());
        let vec_for_index = entry.vector.clone();

        let mut state = self.state.write();

        if state.entries.contains_key(&id) {
            let _ = state.index.remove(&id)?;
        }

        state.index.add(id.clone(), vec_for_index)?;
        state.entries.insert(id.clone(), entry);

        Ok(id)
    }

    fn insert_batch(&self, entries_in: Vec<VectorEntry>) -> Result<Vec<String>, String> {
        let mut prepared = Vec::with_capacity(entries_in.len());
        let mut ids = Vec::with_capacity(entries_in.len());
        let mut index_entries = Vec::with_capacity(entries_in.len());

        for mut entry in entries_in {
            let id = entry
                .id
                .take()
                .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
            entry.id = Some(id.clone());
            index_entries.push((id.clone(), entry.vector.clone()));
            ids.push(id.clone());
            prepared.push((id, entry));
        }

        let mut state = self.state.write();

        for (id, _) in &prepared {
            if state.entries.contains_key(id) {
                let _ = state.index.remove(id)?;
            }
        }

        state.index.add_batch(index_entries)?;
        for (id, entry) in prepared {
            state.entries.insert(id, entry);
        }

        Ok(ids)
    }

    fn get(&self, id: &str) -> Result<Option<VectorEntry>, String> {
        let state = self.state.read();
        Ok(state.entries.get(id).cloned())
    }

    fn delete(&self, id: &str) -> Result<bool, String> {
        let mut state = self.state.write();

        let removed = state.entries.remove(id).is_some();
        if removed {
            let _ = state.index.remove(&id.to_string())?;
        }
        Ok(removed)
    }

    fn len(&self) -> Result<usize, String> {
        Ok(self.state.read().entries.len())
    }

    fn is_empty(&self) -> Result<bool, String> {
        Ok(self.state.read().entries.is_empty())
    }

    fn keys(&self) -> Result<Vec<String>, String> {
        Ok(self.state.read().entries.keys().cloned().collect())
    }

    fn search(&self, query: &SearchQuery, hydrate: bool) -> Result<Vec<SearchResult>, String> {
        if query.vector.len() != self.options.dimensions {
            return Err(format!(
                "Dimension mismatch: expected {}, actual {}",
                self.options.dimensions,
                query.vector.len()
            ));
        }

        if query.k == 0 {
            return Ok(Vec::new());
        }

        let state = self.state.read();
        let total_live = state.entries.len();

        search_with_fallback(
            query.k,
            total_live,
            self.options.hnsw_config.is_some(),
            query.filter.is_some(),
            |fetch_k| {
                let effective_ef = query.ef_search.map(|ef| ef.max(fetch_k));
                let candidates = state.index.search(&query.vector, fetch_k, effective_ef)?;
                let mut seen_ids = HashSet::with_capacity(candidates.len());
                let mut live_count = 0usize;
                let mut matched = Vec::new();

                for mut r in candidates {
                    let Some(entry) = state.entries.get(&r.id) else {
                        continue;
                    };
                    if !seen_ids.insert(r.id.clone()) {
                        continue;
                    }
                    live_count += 1;
                    if let Some(filter) = &query.filter {
                        if !matches_filter(entry.metadata.as_ref(), filter) {
                            continue;
                        }
                    }
                    if hydrate {
                        r.vector = Some(entry.vector.clone());
                        r.metadata = entry.metadata.clone();
                    }
                    matched.push(r);
                }

                Ok(IndexPass {
                    raw_count: live_count,
                    matched,
                })
            },
            || {
                let mut results = Vec::new();
                for (id, entry) in &state.entries {
                    if let Some(filter) = &query.filter {
                        if !matches_filter(entry.metadata.as_ref(), filter) {
                            continue;
                        }
                    }
                    let score =
                        distance(&query.vector, &entry.vector, self.options.distance_metric)
                            .map_err(|e| e.to_string())?;
                    results.push(SearchResult {
                        id: id.clone(),
                        score,
                        vector: hydrate.then(|| entry.vector.clone()),
                        metadata: if hydrate {
                            entry.metadata.clone()
                        } else {
                            None
                        },
                    });
                }
                Ok(finalize_fallback_scan(results, query.k))
            },
        )
    }

    fn options(&self) -> &DbOptions {
        &self.options
    }
}

struct IndexPass {
    raw_count: usize,
    matched: Vec<SearchResult>,
}

fn finalize_fallback_scan(mut results: Vec<SearchResult>, k: usize) -> Vec<SearchResult> {
    results.sort_unstable_by(|a, b| {
        a.score
            .partial_cmp(&b.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    results.truncate(k);
    results
}

fn search_with_fallback<F, G>(
    k: usize,
    total_live: usize,
    has_hnsw: bool,
    has_filter: bool,
    mut run_index: F,
    fallback_scan: G,
) -> Result<Vec<SearchResult>, String>
where
    F: FnMut(usize) -> Result<IndexPass, String>,
    G: FnOnce() -> Result<Vec<SearchResult>, String>,
{
    if k == 0 || total_live == 0 {
        return Ok(Vec::new());
    }

    if !has_hnsw {
        if !has_filter {
            let mut pass = run_index(k)?;
            if pass.matched.len() >= k || pass.matched.len() >= total_live {
                pass.matched.truncate(k);
                return Ok(pass.matched);
            }
            return fallback_scan();
        }

        let initial_k = k.saturating_mul(4).max(64).min(total_live);
        let mut pass = run_index(initial_k)?;
        if pass.matched.len() >= k || initial_k >= total_live || pass.raw_count >= total_live {
            pass.matched.truncate(k);
            return Ok(pass.matched);
        }

        return fallback_scan();
    }

    if !has_filter {
        let mut initial = run_index(k)?;
        if initial.matched.len() >= k || initial.matched.len() >= total_live {
            initial.matched.truncate(k);
            return Ok(initial.matched);
        }
    }

    let max_fetch_k = total_live.saturating_mul(16).max(4096);
    let mut fetch_k = k.saturating_mul(4).max(64).min(max_fetch_k);
    let mut prev_raw_count: Option<usize> = None;

    loop {
        let mut pass = run_index(fetch_k)?;
        if pass.matched.len() >= k
            || (!has_filter && pass.matched.len() >= total_live)
            || pass.raw_count >= total_live
        {
            pass.matched.truncate(k);
            return Ok(pass.matched);
        }

        let plateaued =
            prev_raw_count.is_some_and(|prev| pass.raw_count < fetch_k && pass.raw_count <= prev);
        if plateaued || fetch_k >= max_fetch_k {
            break;
        }

        prev_raw_count = Some(pass.raw_count);
        fetch_k = fetch_k.saturating_mul(2).min(max_fetch_k);
    }

    fallback_scan()
}

fn dedupe_batch_entries(
    entries_in: Vec<VectorEntry>,
    expected_dim: usize,
) -> Result<Vec<VectorEntry>, String> {
    for entry in &entries_in {
        if entry.vector.len() != expected_dim {
            return Err(format!(
                "Dimension mismatch: expected {}, actual {}",
                expected_dim,
                entry.vector.len()
            ));
        }
    }

    let mut last_idx_by_id = HashMap::with_capacity(entries_in.len());
    let mut normalized = Vec::with_capacity(entries_in.len());

    for (idx, mut entry) in entries_in.into_iter().enumerate() {
        let id = entry
            .id
            .take()
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        entry.id = Some(id.clone());
        last_idx_by_id.insert(id.clone(), idx);
        normalized.push((id, entry));
    }

    let mut deduped = Vec::with_capacity(last_idx_by_id.len());
    for (idx, (id, entry)) in normalized.into_iter().enumerate() {
        if last_idx_by_id.get(&id).copied() == Some(idx) {
            deduped.push(entry);
        }
    }

    Ok(deduped)
}

enum DbBackend {
    Persistent(VectorDB),
    InMemory(InMemoryDB),
}

impl DbBackend {
    fn insert(&self, entry: VectorEntry) -> Result<String, String> {
        match self {
            DbBackend::Persistent(db) => db.insert(entry).map_err(|e| e.to_string()),
            DbBackend::InMemory(db) => db.insert(entry),
        }
    }

    fn insert_batch(&self, entries: Vec<VectorEntry>) -> Result<Vec<String>, String> {
        let deduped = dedupe_batch_entries(entries, self.options().dimensions)?;
        match self {
            DbBackend::Persistent(db) => db.insert_batch(deduped).map_err(|e| e.to_string()),
            DbBackend::InMemory(db) => db.insert_batch(deduped),
        }
    }

    fn get(&self, id: &str) -> Result<Option<VectorEntry>, String> {
        match self {
            DbBackend::Persistent(db) => db.get(id).map_err(|e| e.to_string()),
            DbBackend::InMemory(db) => db.get(id),
        }
    }

    fn delete(&self, id: &str) -> Result<bool, String> {
        match self {
            DbBackend::Persistent(db) => db.delete(id).map_err(|e| e.to_string()),
            DbBackend::InMemory(db) => db.delete(id),
        }
    }

    fn len(&self) -> Result<usize, String> {
        match self {
            DbBackend::Persistent(db) => db.len().map_err(|e| e.to_string()),
            DbBackend::InMemory(db) => db.len(),
        }
    }

    fn is_empty(&self) -> Result<bool, String> {
        match self {
            DbBackend::Persistent(db) => db.is_empty().map_err(|e| e.to_string()),
            DbBackend::InMemory(db) => db.is_empty(),
        }
    }

    fn keys(&self) -> Result<Vec<String>, String> {
        match self {
            DbBackend::Persistent(db) => db.keys().map_err(|e| e.to_string()),
            DbBackend::InMemory(db) => db.keys(),
        }
    }

    fn search(&self, query: SearchQuery, hydrate: bool) -> Result<Vec<SearchResult>, String> {
        match self {
            DbBackend::InMemory(db) => db.search(&query, hydrate),
            DbBackend::Persistent(db) => {
                let expected_dim = db.options().dimensions;
                if query.vector.len() != expected_dim {
                    return Err(format!(
                        "Dimension mismatch: expected {}, actual {}",
                        expected_dim,
                        query.vector.len()
                    ));
                }

                if query.k == 0 {
                    return Ok(Vec::new());
                }

                let total_live = db.len().map_err(|e| e.to_string())?;
                let has_hnsw = db.options().hnsw_config.is_some();

                search_with_fallback(
                    query.k,
                    total_live,
                    has_hnsw,
                    query.filter.is_some(),
                    |fetch_k| {
                        let candidates = db
                            .search(SearchQuery {
                                vector: query.vector.clone(),
                                k: fetch_k,
                                filter: None,
                                ef_search: query.ef_search,
                            })
                            .map_err(|e| e.to_string())?;

                        let mut seen_ids = HashSet::with_capacity(candidates.len());
                        let mut live_count = 0usize;
                        let mut matched = Vec::new();

                        for mut r in candidates {
                            // VectorDB::search sets r.vector = Some(...) only for entries present in storage.
                            if r.vector.is_none() {
                                continue;
                            }
                            if !seen_ids.insert(r.id.clone()) {
                                continue;
                            }
                            live_count += 1;
                            if let Some(filter) = &query.filter {
                                if !matches_filter(r.metadata.as_ref(), filter) {
                                    continue;
                                }
                            }
                            if !hydrate {
                                r.vector = None;
                                r.metadata = None;
                            }
                            matched.push(r);
                        }

                        Ok(IndexPass {
                            raw_count: live_count,
                            matched,
                        })
                    },
                    || {
                        let ids = db.keys().map_err(|e| e.to_string())?;
                        let metric = db.options().distance_metric;
                        let mut results = Vec::new();

                        for id in ids {
                            let Some(entry) = db.get(&id).map_err(|e| e.to_string())? else {
                                continue;
                            };
                            if let Some(filter) = &query.filter {
                                if !matches_filter(entry.metadata.as_ref(), filter) {
                                    continue;
                                }
                            }
                            let score = distance(&query.vector, &entry.vector, metric)
                                .map_err(|e| e.to_string())?;
                            results.push(SearchResult {
                                id,
                                score,
                                vector: hydrate.then_some(entry.vector),
                                metadata: if hydrate { entry.metadata } else { None },
                            });
                        }

                        Ok(finalize_fallback_scan(results, query.k))
                    },
                )
            }
        }
    }

    fn options(&self) -> &DbOptions {
        match self {
            DbBackend::Persistent(db) => db.options(),
            DbBackend::InMemory(db) => db.options(),
        }
    }
}

fn json_number_as_i128(n: &serde_json::Number) -> Option<i128> {
    n.as_i64()
        .map(i128::from)
        .or_else(|| n.as_u64().map(i128::from))
}

fn f64_eq_i128(f: f64, i: i128) -> bool {
    const I128_MIN_F64: f64 = -170_141_183_460_469_231_731_687_303_715_884_105_728.0;
    const I128_MAX_BOUND_F64: f64 = 170_141_183_460_469_231_731_687_303_715_884_105_728.0;

    f.is_finite()
        && f.fract() == 0.0
        && (I128_MIN_F64..I128_MAX_BOUND_F64).contains(&f)
        && (f as i128) == i
        && (i as f64) == f
}

fn json_values_equal(a: &serde_json::Value, b: &serde_json::Value) -> bool {
    match (a, b) {
        (serde_json::Value::Number(na), serde_json::Value::Number(nb)) => {
            match (json_number_as_i128(na), json_number_as_i128(nb)) {
                (Some(ia), Some(ib)) => ia == ib,
                (Some(i), None) => nb.as_f64().is_some_and(|f| f64_eq_i128(f, i)),
                (None, Some(i)) => na.as_f64().is_some_and(|f| f64_eq_i128(f, i)),
                (None, None) => match (na.as_f64(), nb.as_f64()) {
                    (Some(fa), Some(fb)) => fa == fb,
                    _ => false,
                },
            }
        }
        (serde_json::Value::Array(aa), serde_json::Value::Array(ab)) => {
            aa.len() == ab.len()
                && aa
                    .iter()
                    .zip(ab.iter())
                    .all(|(xa, xb)| json_values_equal(xa, xb))
        }
        (serde_json::Value::Object(oa), serde_json::Value::Object(ob)) => {
            oa.len() == ob.len()
                && oa
                    .iter()
                    .all(|(k_str, va)| ob.get(k_str).is_some_and(|vb| json_values_equal(va, vb)))
        }
        _ => a == b,
    }
}

fn matches_filter(
    meta: Option<&HashMap<String, serde_json::Value>>,
    filter: &HashMap<String, serde_json::Value>,
) -> bool {
    let Some(meta) = meta else {
        return false;
    };
    filter.iter().all(|(k_str, v_val)| {
        meta.get(k_str)
            .is_some_and(|mv| json_values_equal(mv, v_val))
    })
}

pub struct DbResource {
    db: RwLock<Option<DbBackend>>,
}

static DROP_REAPER: OnceLock<Option<mpsc::Sender<DbBackend>>> = OnceLock::new();

fn drop_reaper_sender() -> Option<&'static mpsc::Sender<DbBackend>> {
    DROP_REAPER
        .get_or_init(|| {
            let (tx, rx) = mpsc::channel::<DbBackend>();
            std::thread::Builder::new()
                .name("ruvector-db-reaper".to_string())
                .spawn(move || {
                    while let Ok(backend) = rx.recv() {
                        drop(backend);
                    }
                })
                .ok()
                .map(|_| tx)
        })
        .as_ref()
}

impl Drop for DbResource {
    fn drop(&mut self) {
        if let Some(backend) = self.db.get_mut().take() {
            // Offload redb close and index deallocation from BEAM GC normal schedulers.
            if let Some(tx) = drop_reaper_sender() {
                if let Err(mpsc::SendError(backend)) = tx.send(backend) {
                    drop(backend);
                }
            } else {
                drop(backend);
            }
        }
    }
}

#[rustler::resource_impl]
impl rustler::Resource for DbResource {}

rustler::atoms! {
    ok,
    error,
    not_found,
    closed,
    cosine,
    euclidean,
    dot_product,
    manhattan,
    none,
    scalar,
    binary,
    product,
    type_ = "type",
    subspaces,
    k,
    id,
    vector,
    metadata,
    score,
    node_count,
    dimensions,
    distance_metric,
    storage_path,
    hnsw,
    quantization,
    m,
    ef_construction,
    ef_search,
    max_elements,
}

enum DbAccessError {
    Closed,
    Message(String),
}

impl DbAccessError {
    fn encode<'a>(&self, env: Env<'a>) -> Term<'a> {
        match self {
            DbAccessError::Closed => (error(), closed()).encode(env),
            DbAccessError::Message(msg) => (error(), msg.as_str()).encode(env),
        }
    }
}

fn with_db<F, R>(resource: &ResourceArc<DbResource>, f: F) -> Result<R, DbAccessError>
where
    F: FnOnce(&DbBackend) -> Result<R, String>,
{
    let guard = resource.db.read();
    match guard.as_ref() {
        Some(backend) => f(backend).map_err(DbAccessError::Message),
        None => Err(DbAccessError::Closed),
    }
}

fn is_nil_or_bool(term: Term) -> bool {
    rustler::types::atom::nil() == term || term.decode::<bool>().is_ok()
}

fn decode_map_key(term: Term) -> Result<String, rustler::Error> {
    if let Ok(s) = term.decode::<String>() {
        Ok(s)
    } else if is_nil_or_bool(term) {
        Err(rustler::Error::BadArg)
    } else if let Ok(name) = term.atom_to_string() {
        if name == "__struct__" {
            Err(rustler::Error::BadArg)
        } else {
            Ok(name)
        }
    } else if let Ok(i) = term.decode::<i64>() {
        Ok(i.to_string())
    } else if let Ok(u) = term.decode::<u64>() {
        Ok(u.to_string())
    } else {
        Err(rustler::Error::BadArg)
    }
}

fn map_term_to_json_map(
    term: Term,
    depth: usize,
) -> Result<HashMap<String, serde_json::Value>, rustler::Error> {
    if depth > MAX_JSON_DEPTH {
        return Err(rustler::Error::BadArg);
    }

    let map_iter = rustler::types::map::MapIterator::new(term).ok_or(rustler::Error::BadArg)?;
    let mut map = HashMap::new();

    for (k_term, v_term) in map_iter {
        if rustler::types::atom::__struct__() == k_term {
            return Err(rustler::Error::BadArg);
        }
        let key = decode_map_key(k_term)?;
        let val = term_to_json_depth(v_term, depth + 1)?;
        if map.insert(key, val).is_some() {
            return Err(rustler::Error::BadArg);
        }
    }

    Ok(map)
}

fn term_to_json(term: Term) -> Result<serde_json::Value, rustler::Error> {
    term_to_json_depth(term, 0)
}

// Strictly convert a supported Elixir term into a serde_json::Value with a
// recursion depth cap to prevent OS scheduler-thread stack overflow.
fn term_to_json_depth(term: Term, depth: usize) -> Result<serde_json::Value, rustler::Error> {
    if depth > MAX_JSON_DEPTH {
        return Err(rustler::Error::BadArg);
    }

    if rustler::types::atom::nil() == term {
        return Ok(serde_json::Value::Null);
    }
    if let Ok(b) = term.decode::<bool>() {
        return Ok(serde_json::Value::Bool(b));
    }
    if term.is_float() {
        let f = term.decode::<f64>()?;
        if f.is_finite() {
            if let Some(n) = serde_json::Number::from_f64(f) {
                return Ok(serde_json::Value::Number(n));
            }
        }
        return Err(rustler::Error::BadArg);
    }
    if let Ok(i) = term.decode::<i64>() {
        return Ok(serde_json::Value::Number(i.into()));
    }
    if let Ok(u) = term.decode::<u64>() {
        return Ok(serde_json::Value::Number(u.into()));
    }
    if let Ok(s) = term.decode::<String>() {
        return Ok(serde_json::Value::String(s));
    }
    if let Ok(name) = term.atom_to_string() {
        return Ok(serde_json::Value::String(name));
    }
    if term.is_list() {
        let list = term.decode::<Vec<Term>>()?;
        let mut values = Vec::with_capacity(list.len());
        for item in list {
            values.push(term_to_json_depth(item, depth + 1)?);
        }
        return Ok(serde_json::Value::Array(values));
    }
    if let Some(map_iter) = rustler::types::map::MapIterator::new(term) {
        let mut map = serde_json::Map::new();
        for (k_term, v_term) in map_iter {
            if rustler::types::atom::__struct__() == k_term {
                return Err(rustler::Error::BadArg);
            }
            let key = decode_map_key(k_term)?;
            let val = term_to_json_depth(v_term, depth + 1)?;
            if map.insert(key, val).is_some() {
                return Err(rustler::Error::BadArg);
            }
        }
        return Ok(serde_json::Value::Object(map));
    }

    Err(rustler::Error::BadArg)
}

fn json_to_term<'a>(env: Env<'a>, val: &serde_json::Value, depth: usize) -> NifResult<Term<'a>> {
    if depth > MAX_JSON_DEPTH {
        return Err(rustler::Error::BadArg);
    }

    match val {
        serde_json::Value::Null => Ok(rustler::types::atom::nil().encode(env)),
        serde_json::Value::Bool(b) => Ok(b.encode(env)),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(i.encode(env))
            } else if let Some(u) = n.as_u64() {
                Ok(u.encode(env))
            } else if let Some(f) = n.as_f64() {
                if f.is_finite() {
                    Ok(f.encode(env))
                } else {
                    Err(rustler::Error::BadArg)
                }
            } else {
                Ok(n.to_string().encode(env))
            }
        }
        serde_json::Value::String(s) => Ok(s.encode(env)),
        serde_json::Value::Array(arr) => {
            let mut terms = Vec::with_capacity(arr.len());
            for v in arr {
                terms.push(json_to_term(env, v, depth + 1)?);
            }
            Ok(terms.encode(env))
        }
        serde_json::Value::Object(obj) => {
            let mut keys = Vec::with_capacity(obj.len());
            let mut vals = Vec::with_capacity(obj.len());
            for (k_str, v) in obj {
                keys.push(k_str.encode(env));
                vals.push(json_to_term(env, v, depth + 1)?);
            }
            Term::map_from_term_arrays(env, &keys, &vals)
        }
    }
}

fn metadata_to_term<'a>(
    env: Env<'a>,
    meta: Option<&HashMap<String, serde_json::Value>>,
) -> Result<Term<'a>, &'static str> {
    match meta {
        None => Ok(rustler::types::atom::nil().encode(env)),
        Some(map) => {
            let mut keys = Vec::with_capacity(map.len());
            let mut vals = Vec::with_capacity(map.len());
            for (k_str, v) in map {
                keys.push(k_str.encode(env));
                vals.push(json_to_term(env, v, 1).map_err(|_| "Invalid stored metadata")?);
            }
            Term::map_from_term_arrays(env, &keys, &vals).map_err(|_| "Invalid stored metadata")
        }
    }
}

fn decode_finite_f32_vec(term: Term) -> Result<Vec<f32>, rustler::Error> {
    let vec = term.decode::<Vec<f32>>()?;
    if vec.iter().all(|x| x.is_finite()) {
        Ok(vec)
    } else {
        Err(rustler::Error::BadArg)
    }
}

fn decode_id(term: Term) -> Result<String, rustler::Error> {
    if let Ok(s) = term.decode::<String>() {
        if s.is_empty() {
            Err(rustler::Error::BadArg)
        } else {
            Ok(s)
        }
    } else if is_nil_or_bool(term) {
        Err(rustler::Error::BadArg)
    } else if let Ok(name) = term.atom_to_string() {
        if name.is_empty() {
            Err(rustler::Error::BadArg)
        } else {
            Ok(name)
        }
    } else if let Ok(i) = term.decode::<i64>() {
        Ok(i.to_string())
    } else if let Ok(u) = term.decode::<u64>() {
        Ok(u.to_string())
    } else if let Ok(i) = term.decode::<i128>() {
        Ok(i.to_string())
    } else if let Ok(u) = term.decode::<u128>() {
        Ok(u.to_string())
    } else {
        Err(rustler::Error::BadArg)
    }
}

fn decode_vector_entry(term: Term) -> Result<VectorEntry, rustler::Error> {
    if term.is_list() {
        let vec = decode_finite_f32_vec(term)?;
        return Ok(VectorEntry {
            id: None,
            vector: vec,
            metadata: None,
        });
    }

    if let Some(map_iter) = rustler::types::map::MapIterator::new(term) {
        let mut id_val: Option<String> = None;
        let mut vector_val: Option<Vec<f32>> = None;
        let mut metadata_val: Option<HashMap<String, serde_json::Value>> = None;

        let mut seen_id = false;
        let mut seen_vector = false;
        let mut seen_metadata = false;

        for (k_term, v_term) in map_iter {
            let key: String = if let Ok(s) = k_term.decode::<String>() {
                s
            } else if let Ok(name) = k_term.atom_to_string() {
                if name == "__struct__" {
                    if v_term.atom_to_string().ok().as_deref()
                        != Some("Elixir.RuvectorElixir.VectorEntry")
                    {
                        return Err(rustler::Error::BadArg);
                    }
                    continue;
                }
                name
            } else {
                return Err(rustler::Error::BadArg);
            };

            match key.as_str() {
                "id" => {
                    if seen_id {
                        return Err(rustler::Error::BadArg);
                    }
                    seen_id = true;
                    if rustler::types::atom::nil() == v_term {
                        id_val = None;
                    } else {
                        id_val = Some(decode_id(v_term)?);
                    }
                }
                "vector" => {
                    if seen_vector {
                        return Err(rustler::Error::BadArg);
                    }
                    seen_vector = true;
                    vector_val = Some(decode_finite_f32_vec(v_term)?);
                }
                "metadata" => {
                    if seen_metadata {
                        return Err(rustler::Error::BadArg);
                    }
                    seen_metadata = true;
                    if rustler::types::atom::nil() == v_term {
                        metadata_val = None;
                    } else {
                        metadata_val = Some(map_term_to_json_map(v_term, 1)?);
                    }
                }
                _ => return Err(rustler::Error::BadArg),
            }
        }

        if let Some(vec_data) = vector_val {
            return Ok(VectorEntry {
                id: id_val,
                vector: vec_data,
                metadata: metadata_val,
            });
        }
    }

    Err(rustler::Error::BadArg)
}

fn parse_metric_str(s: &str) -> Result<DistanceMetric, String> {
    match s.to_lowercase().as_str() {
        "euclidean" => Ok(DistanceMetric::Euclidean),
        "cosine" => Ok(DistanceMetric::Cosine),
        "dot_product" | "dotproduct" | "dot" => Ok(DistanceMetric::DotProduct),
        "manhattan" => Ok(DistanceMetric::Manhattan),
        _ => Err(format!("Unsupported distance metric: {}", s)),
    }
}

fn parse_metric(term: Term) -> Result<DistanceMetric, String> {
    if is_nil_or_bool(term) {
        return Err("Invalid distance metric".to_string());
    }
    if let Ok(s) = term.atom_to_string() {
        parse_metric_str(&s)
    } else if let Ok(s) = term.decode::<String>() {
        parse_metric_str(&s)
    } else {
        Err("Invalid distance metric".to_string())
    }
}

fn decode_pos_usize(term: Term, field: &str) -> Result<usize, String> {
    match term.decode::<usize>() {
        Ok(val) if val > 0 => Ok(val),
        _ => Err(format!("Option :{} must be a positive integer", field)),
    }
}

fn parse_quantization(term: Term, dimensions: usize) -> Result<Option<QuantizationConfig>, String> {
    if rustler::types::atom::nil() == term {
        return Ok(None);
    }

    if term.decode::<bool>().is_ok() {
        return Err("Invalid :quantization option".to_string());
    }

    let q_str = if let Ok(s) = term.atom_to_string() {
        Some(s)
    } else {
        term.decode::<String>().ok()
    };

    if let Some(s) = q_str {
        return match s.to_lowercase().as_str() {
            "scalar" => Ok(Some(QuantizationConfig::Scalar)),
            "binary" => Ok(Some(QuantizationConfig::Binary)),
            "none" => Ok(Some(QuantizationConfig::None)),
            _ => Err(format!("Unsupported :quantization option: {}", s)),
        };
    }

    if let Some(map_iter) = rustler::types::map::MapIterator::new(term) {
        let mut q_type: Option<String> = None;
        let mut subspaces_val: Option<usize> = None;
        let mut k_val: Option<usize> = None;

        for (k_term, v_term) in map_iter {
            let key = decode_map_key(k_term)
                .map_err(|_| "Invalid key in :quantization option".to_string())?;

            match key.as_str() {
                "type" => {
                    if q_type.is_some() {
                        return Err("Duplicate :type in :quantization option".to_string());
                    }
                    if is_nil_or_bool(v_term) {
                        return Err("Invalid :type in :quantization option".to_string());
                    }
                    let t_str = if let Ok(s) = v_term.decode::<String>() {
                        s
                    } else if let Ok(name) = v_term.atom_to_string() {
                        name
                    } else {
                        return Err("Invalid :type in :quantization option".to_string());
                    };
                    q_type = Some(t_str.to_lowercase());
                }
                "subspaces" => {
                    if subspaces_val.is_some() {
                        return Err("Duplicate :subspaces in :quantization option".to_string());
                    }
                    subspaces_val = Some(decode_pos_usize(v_term, "subspaces")?);
                }
                "k" => {
                    if k_val.is_some() {
                        return Err("Duplicate :k in :quantization option".to_string());
                    }
                    k_val = Some(decode_pos_usize(v_term, "k")?);
                }
                other => {
                    return Err(format!("Unknown :quantization option key: {}", other));
                }
            }
        }

        let build_product =
            |sub: Option<usize>, k_opt: Option<usize>| -> Result<QuantizationConfig, String> {
                let s = sub.unwrap_or(8);
                if !dimensions.is_multiple_of(s) {
                    return Err(format!(
                        ":quantization :subspaces ({}) must evenly divide dimensions ({})",
                        s, dimensions
                    ));
                }
                Ok(QuantizationConfig::Product {
                    subspaces: s,
                    k: k_opt.unwrap_or(256),
                })
            };

        return match q_type.as_deref() {
            Some("product") => Ok(Some(build_product(subspaces_val, k_val)?)),
            None if subspaces_val.is_some() || k_val.is_some() => {
                Ok(Some(build_product(subspaces_val, k_val)?))
            }
            Some("scalar") if subspaces_val.is_none() && k_val.is_none() => {
                Ok(Some(QuantizationConfig::Scalar))
            }
            Some("binary") if subspaces_val.is_none() && k_val.is_none() => {
                Ok(Some(QuantizationConfig::Binary))
            }
            Some("none") if subspaces_val.is_none() && k_val.is_none() => {
                Ok(Some(QuantizationConfig::None))
            }
            Some(other) if matches!(other, "scalar" | "binary" | "none") => Err(format!(
                ":subspaces and :k are only valid with :product quantization, not :{}",
                other
            )),
            Some(other) => Err(format!("Unsupported quantization type: {}", other)),
            None => Err("Empty :quantization option map".to_string()),
        };
    }

    Err("Invalid :quantization option".to_string())
}

fn parse_hnsw(term: Term) -> Result<Option<HnswConfig>, String> {
    if rustler::types::atom::nil() == term {
        return Ok(None);
    }
    if let Ok(b) = term.decode::<bool>() {
        return if b {
            Ok(Some(HnswConfig::default()))
        } else {
            Ok(None)
        };
    }
    if let Some(hnsw_map) = rustler::types::map::MapIterator::new(term) {
        let mut cfg = HnswConfig::default();
        let mut count = 0usize;
        for (hk, hv) in hnsw_map {
            count += 1;
            let hkey = decode_map_key(hk).map_err(|_| "Invalid key in :hnsw option".to_string())?;
            match hkey.as_str() {
                "m" => {
                    let val = decode_pos_usize(hv, "m")?;
                    if !(2..=256).contains(&val) {
                        return Err(format!("Option :m must be between 2 and 256, got: {}", val));
                    }
                    cfg.m = val;
                }
                "ef_construction" => cfg.ef_construction = decode_pos_usize(hv, "ef_construction")?,
                "ef_search" => cfg.ef_search = decode_pos_usize(hv, "ef_search")?,
                "max_elements" => cfg.max_elements = decode_pos_usize(hv, "max_elements")?,
                other => return Err(format!("Unknown :hnsw option key: {}", other)),
            }
        }
        if count == 0 {
            return Err("Empty :hnsw option map".to_string());
        }
        return Ok(Some(cfg));
    }

    Err("Invalid :hnsw option".to_string())
}

fn parse_db_options(
    path: String,
    dimensions: usize,
    opts_term: Option<Term>,
) -> Result<DbOptions, String> {
    if path.is_empty() {
        return Err("Database path cannot be empty".to_string());
    }
    if dimensions == 0 {
        return Err("Dimensions must be a positive integer".to_string());
    }

    let mut distance_metric = DistanceMetric::Cosine;
    let mut seen_metric = false;
    let mut hnsw_config = None;
    let mut quantization = None;

    if let Some(opts) = opts_term {
        if rustler::types::atom::nil() != opts {
            let map_iter = rustler::types::map::MapIterator::new(opts)
                .ok_or_else(|| "Database options must be a map".to_string())?;

            for (k_term, v_term) in map_iter {
                let key = decode_map_key(k_term).map_err(|_| "Invalid option key".to_string())?;

                match key.as_str() {
                    "metric" | "distance_metric" => {
                        if seen_metric {
                            return Err(
                                "Cannot specify both :metric and :distance_metric".to_string()
                            );
                        }
                        seen_metric = true;
                        distance_metric = parse_metric(v_term)?;
                    }
                    "hnsw" => {
                        hnsw_config = parse_hnsw(v_term)?;
                    }
                    "quantization" => {
                        quantization = parse_quantization(v_term, dimensions)?;
                    }
                    other => {
                        return Err(format!("Unknown database option: {}", other));
                    }
                }
            }
        }
    }

    Ok(DbOptions {
        dimensions,
        distance_metric,
        storage_path: path,
        hnsw_config,
        quantization,
    })
}

#[rustler::nif(schedule = "DirtyIo")]
fn open_db_with_options<'a>(
    env: Env<'a>,
    path_term: Term<'a>,
    dimensions_term: Term<'a>,
    options: Option<Term<'a>>,
) -> NifResult<Term<'a>> {
    let Ok(path) = path_term.decode::<String>() else {
        return Ok((error(), "Database path must be a valid UTF-8 string").encode(env));
    };
    let Ok(dimensions) = dimensions_term.decode::<usize>() else {
        return Ok((error(), "Dimensions must be a positive integer").encode(env));
    };

    let opts = match parse_db_options(path, dimensions, options) {
        Ok(o) => o,
        Err(msg) => return Ok((error(), msg).encode(env)),
    };

    let backend_res = if opts.storage_path.starts_with("memory://") {
        InMemoryDB::new(opts).map(DbBackend::InMemory)
    } else {
        VectorDB::new(opts)
            .map(DbBackend::Persistent)
            .map_err(|e| e.to_string())
    };

    match backend_res {
        Ok(backend) => {
            let resource = ResourceArc::new(DbResource {
                db: RwLock::new(Some(backend)),
            });
            Ok((ok(), resource).encode(env))
        }
        Err(err) => Ok((error(), err).encode(env)),
    }
}

#[rustler::nif(schedule = "DirtyIo")]
fn close_db<'a>(env: Env<'a>, db_ref: ResourceArc<DbResource>) -> NifResult<Term<'a>> {
    let backend = db_ref.db.write().take();
    drop(backend);
    Ok(ok().encode(env))
}

#[rustler::nif(schedule = "DirtyIo")]
fn insert<'a>(
    env: Env<'a>,
    db_ref: ResourceArc<DbResource>,
    entry_term: Term<'a>,
) -> NifResult<Term<'a>> {
    let entry = match decode_vector_entry(entry_term) {
        Ok(e) => e,
        Err(_) => return Ok((error(), "Invalid vector entry format").encode(env)),
    };

    match with_db(&db_ref, |db| db.insert(entry)) {
        Ok(id_str) => Ok((ok(), id_str).encode(env)),
        Err(err) => Ok(err.encode(env)),
    }
}

#[rustler::nif(schedule = "DirtyIo")]
fn insert_batch<'a>(
    env: Env<'a>,
    db_ref: ResourceArc<DbResource>,
    entries_term: Term<'a>,
) -> NifResult<Term<'a>> {
    let Ok(entries_list) = entries_term.decode::<Vec<Term<'a>>>() else {
        return Ok((error(), "Vector batch must be a list").encode(env));
    };

    let mut entries = Vec::with_capacity(entries_list.len());
    for t in entries_list {
        match decode_vector_entry(t) {
            Ok(e) => entries.push(e),
            Err(_) => return Ok((error(), "Invalid vector entry in batch").encode(env)),
        }
    }

    match with_db(&db_ref, |db| db.insert_batch(entries)) {
        Ok(ids) => Ok((ok(), ids).encode(env)),
        Err(err) => Ok(err.encode(env)),
    }
}

#[rustler::nif(schedule = "DirtyIo")]
fn get_entry<'a>(
    env: Env<'a>,
    db_ref: ResourceArc<DbResource>,
    id_term: Term<'a>,
) -> NifResult<Term<'a>> {
    let id_str = match decode_id(id_term) {
        Ok(s) => s,
        Err(_) => return Ok((error(), "Invalid id format").encode(env)),
    };

    match with_db(&db_ref, |db| db.get(&id_str)) {
        Ok(Some(entry)) => {
            if !entry.vector.iter().all(|x| x.is_finite()) {
                return Ok((error(), "Stored vector contains non-finite float").encode(env));
            }
            let entry_id = entry.id.as_deref().unwrap_or(&id_str);
            let meta_term = match metadata_to_term(env, entry.metadata.as_ref()) {
                Ok(t) => t,
                Err(msg) => return Ok((error(), msg).encode(env)),
            };
            let map = rustler::term_map!(env, {
                id() => entry_id,
                vector() => entry.vector,
                metadata() => meta_term,
            });
            Ok((ok(), map).encode(env))
        }
        Ok(None) => Ok((error(), not_found()).encode(env)),
        Err(err) => Ok(err.encode(env)),
    }
}

#[rustler::nif(schedule = "DirtyIo")]
fn delete<'a>(
    env: Env<'a>,
    db_ref: ResourceArc<DbResource>,
    id_term: Term<'a>,
) -> NifResult<Term<'a>> {
    let id_str = match decode_id(id_term) {
        Ok(s) => s,
        Err(_) => return Ok((error(), "Invalid id format").encode(env)),
    };

    match with_db(&db_ref, |db| db.delete(&id_str)) {
        Ok(deleted) => Ok((ok(), deleted).encode(env)),
        Err(err) => Ok(err.encode(env)),
    }
}

#[rustler::nif(schedule = "DirtyIo")]
fn db_len<'a>(env: Env<'a>, db_ref: ResourceArc<DbResource>) -> NifResult<Term<'a>> {
    match with_db(&db_ref, |db| db.len()) {
        Ok(count) => Ok((ok(), count).encode(env)),
        Err(err) => Ok(err.encode(env)),
    }
}

#[rustler::nif(schedule = "DirtyIo")]
fn is_empty<'a>(env: Env<'a>, db_ref: ResourceArc<DbResource>) -> NifResult<Term<'a>> {
    match with_db(&db_ref, |db| db.is_empty()) {
        Ok(empty) => Ok((ok(), empty).encode(env)),
        Err(err) => Ok(err.encode(env)),
    }
}

#[rustler::nif(schedule = "DirtyIo")]
fn all_ids<'a>(env: Env<'a>, db_ref: ResourceArc<DbResource>) -> NifResult<Term<'a>> {
    match with_db(&db_ref, |db| db.keys()) {
        Ok(ids) => Ok((ok(), ids).encode(env)),
        Err(err) => Ok(err.encode(env)),
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum FilterKeyKind {
    String,
    Atom,
}

fn extract_filter<'a>(
    filter: Option<Term<'a>>,
) -> Result<Option<HashMap<String, serde_json::Value>>, rustler::Error> {
    let Some(f_term) = filter else {
        return Ok(None);
    };

    if rustler::types::atom::nil() == f_term {
        return Ok(None);
    }

    if rustler::types::map::MapIterator::new(f_term).is_some() {
        let map = map_term_to_json_map(f_term, 1)?;
        return if map.is_empty() {
            Ok(None)
        } else {
            Ok(Some(map))
        };
    }

    if f_term.is_list() {
        let list = f_term.decode::<Vec<(Term, Term)>>()?;
        if list.is_empty() {
            return Ok(None);
        }
        let mut map = HashMap::with_capacity(list.len());
        let mut key_kinds: HashMap<String, FilterKeyKind> = HashMap::with_capacity(list.len());
        for (k_term, v_term) in list {
            let (key, kind) = if let Ok(s) = k_term.decode::<String>() {
                (s, FilterKeyKind::String)
            } else if is_nil_or_bool(k_term) {
                return Err(rustler::Error::BadArg);
            } else if let Ok(name) = k_term.atom_to_string() {
                if name == "__struct__" {
                    return Err(rustler::Error::BadArg);
                }
                (name, FilterKeyKind::Atom)
            } else {
                return Err(rustler::Error::BadArg);
            };
            if let Some(&prev_kind) = key_kinds.get(&key) {
                if prev_kind != kind {
                    return Err(rustler::Error::BadArg);
                }
            } else {
                key_kinds.insert(key.clone(), kind);
            }
            let val = term_to_json(v_term)?;
            // Match Elixir Keyword first-key-wins semantics on duplicate same-kind keys.
            map.entry(key).or_insert(val);
        }
        return Ok(Some(map));
    }

    Err(rustler::Error::BadArg)
}

fn decode_search_query<'a>(
    query_vector_term: Term<'a>,
    filter: Option<Term<'a>>,
    limit_term: Term<'a>,
    ef_search_term: Option<Term<'a>>,
) -> Result<SearchQuery, String> {
    let query_vector =
        decode_finite_f32_vec(query_vector_term).map_err(|_| "Invalid query vector".to_string())?;
    let limit = limit_term
        .decode::<usize>()
        .map_err(|_| "Invalid search limit k".to_string())?;
    let ef_search_arg = match ef_search_term {
        None => None,
        Some(t) if rustler::types::atom::nil() == t => None,
        Some(t) => Some(decode_pos_usize(t, "ef_search")?),
    };
    let query_filter = extract_filter(filter).map_err(|_| "Invalid filter format".to_string())?;

    Ok(SearchQuery {
        vector: query_vector,
        k: limit,
        filter: query_filter,
        ef_search: ef_search_arg,
    })
}

#[rustler::nif(schedule = "DirtyCpu")]
fn search<'a>(
    env: Env<'a>,
    db_ref: ResourceArc<DbResource>,
    query_vector_term: Term<'a>,
    filter: Option<Term<'a>>,
    limit_term: Term<'a>,
    ef_search_term: Option<Term<'a>>,
) -> NifResult<Term<'a>> {
    let query = match decode_search_query(query_vector_term, filter, limit_term, ef_search_term) {
        Ok(q) => q,
        Err(msg) => return Ok((error(), msg).encode(env)),
    };

    match with_db(&db_ref, |db| db.search(query, false)) {
        Ok(results) => {
            if results.iter().any(|r| !r.score.is_finite()) {
                return Ok((error(), "Search produced non-finite score").encode(env));
            }
            let ids: Vec<String> = results.into_iter().map(|r| r.id).collect();
            Ok((ok(), ids).encode(env))
        }
        Err(err) => Ok(err.encode(env)),
    }
}

#[rustler::nif(schedule = "DirtyCpu")]
fn search_detailed<'a>(
    env: Env<'a>,
    db_ref: ResourceArc<DbResource>,
    query_vector_term: Term<'a>,
    filter: Option<Term<'a>>,
    limit_term: Term<'a>,
    ef_search_term: Option<Term<'a>>,
) -> NifResult<Term<'a>> {
    let query = match decode_search_query(query_vector_term, filter, limit_term, ef_search_term) {
        Ok(q) => q,
        Err(msg) => return Ok((error(), msg).encode(env)),
    };

    match with_db(&db_ref, |db| db.search(query, true)) {
        Ok(results) => {
            let mut terms = Vec::with_capacity(results.len());
            for r in results {
                if !r.score.is_finite() {
                    return Ok((error(), "Search produced non-finite score").encode(env));
                }
                let vec_term = match r.vector {
                    Some(v) => {
                        if !v.iter().all(|x| x.is_finite()) {
                            return Ok(
                                (error(), "Stored vector contains non-finite float").encode(env)
                            );
                        }
                        v.encode(env)
                    }
                    None => rustler::types::atom::nil().encode(env),
                };
                let meta_term = match metadata_to_term(env, r.metadata.as_ref()) {
                    Ok(t) => t,
                    Err(msg) => return Ok((error(), msg).encode(env)),
                };
                let map = rustler::term_map!(env, {
                    id() => r.id,
                    score() => r.score,
                    vector() => vec_term,
                    metadata() => meta_term,
                });
                terms.push(map);
            }
            Ok((ok(), terms).encode(env))
        }
        Err(err) => Ok(err.encode(env)),
    }
}

fn calculate_distance_impl<'a>(
    env: Env<'a>,
    a_term: Term<'a>,
    b_term: Term<'a>,
    metric_term: Term<'a>,
) -> NifResult<Term<'a>> {
    let a = match decode_finite_f32_vec(a_term) {
        Ok(v) => v,
        Err(_) => return Ok((error(), "Invalid vector a").encode(env)),
    };
    let b = match decode_finite_f32_vec(b_term) {
        Ok(v) => v,
        Err(_) => return Ok((error(), "Invalid vector b").encode(env)),
    };
    let metric = match parse_metric(metric_term) {
        Ok(m) => m,
        Err(msg) => return Ok((error(), msg).encode(env)),
    };

    match ruvector_core::distance::distance(&a, &b, metric) {
        Ok(dist) if dist.is_finite() => Ok((ok(), dist).encode(env)),
        Ok(_) => Ok((error(), "Distance calculation produced non-finite value").encode(env)),
        Err(err) => Ok((error(), err.to_string()).encode(env)),
    }
}

#[rustler::nif]
fn calculate_distance<'a>(
    env: Env<'a>,
    a_term: Term<'a>,
    b_term: Term<'a>,
    metric_term: Term<'a>,
) -> NifResult<Term<'a>> {
    calculate_distance_impl(env, a_term, b_term, metric_term)
}

#[rustler::nif(schedule = "DirtyCpu")]
fn calculate_distance_dirty<'a>(
    env: Env<'a>,
    a_term: Term<'a>,
    b_term: Term<'a>,
    metric_term: Term<'a>,
) -> NifResult<Term<'a>> {
    calculate_distance_impl(env, a_term, b_term, metric_term)
}

fn encode_metric_atom<'a>(env: Env<'a>, metric: DistanceMetric) -> Term<'a> {
    match metric {
        DistanceMetric::Euclidean => euclidean().encode(env),
        DistanceMetric::Cosine => cosine().encode(env),
        DistanceMetric::DotProduct => dot_product().encode(env),
        DistanceMetric::Manhattan => manhattan().encode(env),
    }
}

fn encode_quantization_term<'a>(env: Env<'a>, quant: Option<&QuantizationConfig>) -> Term<'a> {
    match quant {
        None => rustler::types::atom::nil().encode(env),
        Some(QuantizationConfig::None) => none().encode(env),
        Some(QuantizationConfig::Scalar) => scalar().encode(env),
        Some(QuantizationConfig::Binary) => binary().encode(env),
        Some(QuantizationConfig::Product {
            subspaces: sub,
            k: k_size,
        }) => rustler::term_map!(env, {
            type_() => product(),
            subspaces() => *sub,
            k() => *k_size,
        }),
    }
}

#[rustler::nif(schedule = "DirtyIo")]
fn info<'a>(env: Env<'a>, db_ref: ResourceArc<DbResource>) -> NifResult<Term<'a>> {
    let snapshot = with_db(&db_ref, |db| {
        let count = db.len()?;
        let opts = db.options().clone();
        Ok((count, opts))
    });

    match snapshot {
        Ok((count, opts)) => {
            let hnsw_term = match &opts.hnsw_config {
                Some(cfg) => rustler::term_map!(env, {
                    m() => cfg.m,
                    ef_construction() => cfg.ef_construction,
                    ef_search() => cfg.ef_search,
                    max_elements() => cfg.max_elements,
                }),
                None => rustler::types::atom::nil().encode(env),
            };

            let map = rustler::term_map!(env, {
                node_count() => count,
                dimensions() => opts.dimensions,
                distance_metric() => encode_metric_atom(env, opts.distance_metric),
                storage_path() => opts.storage_path,
                hnsw() => hnsw_term,
                quantization() => encode_quantization_term(env, opts.quantization.as_ref()),
            });

            Ok((ok(), map).encode(env))
        }
        Err(err) => Ok(err.encode(env)),
    }
}

rustler::init!("Elixir.RuvectorElixir.Native");
