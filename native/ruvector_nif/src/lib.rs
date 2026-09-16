use rustler::{Encoder, Env, NifResult, ResourceArc, Term};
use ruvector_core::types::{
    DbOptions, DistanceMetric, HnswConfig, QuantizationConfig, SearchQuery, VectorEntry,
};
use ruvector_core::VectorDB;
use std::collections::HashMap;

// Wrap the VectorDB in a struct so we can implement the Resource trait
pub struct DbResource {
    pub db: VectorDB,
}

rustler::atoms! {
    ok,
    error,
    not_found,
    cosine,
    euclidean,
    dot_product,
    manhattan,
    id,
    vector,
    metadata,
    score,
}

#[allow(non_local_definitions)]
fn load(env: Env, _info: Term) -> bool {
    let _ = rustler::resource!(DbResource, env);
    true
}

// Convert an Elixir Term into a serde_json::Value
fn term_to_json(term: Term) -> Result<serde_json::Value, rustler::Error> {
    if rustler::types::atom::nil() == term {
        return Ok(serde_json::Value::Null);
    }
    if let Ok(b) = term.decode::<bool>() {
        return Ok(serde_json::Value::Bool(b));
    }
    if let Ok(i) = term.decode::<i64>() {
        return Ok(serde_json::Value::Number(i.into()));
    }
    if let Ok(u) = term.decode::<u64>() {
        return Ok(serde_json::Value::Number(u.into()));
    }
    if let Ok(f) = term.decode::<f64>() {
        if let Some(n) = serde_json::Number::from_f64(f) {
            return Ok(serde_json::Value::Number(n));
        }
    }
    if let Ok(s) = term.decode::<String>() {
        return Ok(serde_json::Value::String(s));
    }
    if let Ok(name) = term.atom_to_string() {
        return Ok(serde_json::Value::String(name));
    }
    if let Ok(list) = term.decode::<Vec<Term>>() {
        let mut values = Vec::with_capacity(list.len());
        for item in list {
            values.push(term_to_json(item)?);
        }
        return Ok(serde_json::Value::Array(values));
    }
    if let Some(map_iter) = rustler::types::map::MapIterator::new(term) {
        let mut map = serde_json::Map::new();
        for (k_term, v_term) in map_iter {
            let key: String = if let Ok(s) = k_term.decode::<String>() {
                s
            } else if let Ok(name) = k_term.atom_to_string() {
                if name == "__struct__" {
                    continue;
                }
                name
            } else if let Ok(i) = k_term.decode::<i64>() {
                i.to_string()
            } else {
                continue;
            };
            map.insert(key, term_to_json(v_term)?);
        }
        return Ok(serde_json::Value::Object(map));
    }
    Ok(serde_json::Value::Null)
}

// Convert a serde_json::Value back into an Elixir Term
fn json_to_term<'a>(env: Env<'a>, val: &serde_json::Value) -> Term<'a> {
    match val {
        serde_json::Value::Null => rustler::types::atom::nil().encode(env),
        serde_json::Value::Bool(b) => b.encode(env),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                i.encode(env)
            } else if let Some(u) = n.as_u64() {
                u.encode(env)
            } else if let Some(f) = n.as_f64() {
                f.encode(env)
            } else {
                n.to_string().encode(env)
            }
        }
        serde_json::Value::String(s) => s.encode(env),
        serde_json::Value::Array(arr) => {
            let terms: Vec<Term<'a>> = arr.iter().map(|v| json_to_term(env, v)).collect();
            terms.encode(env)
        }
        serde_json::Value::Object(obj) => {
            let mut map = Term::map_new(env);
            for (k, v) in obj {
                map = map.map_put(k.encode(env), json_to_term(env, v)).unwrap();
            }
            map
        }
    }
}

fn decode_id(term: Term) -> Result<String, rustler::Error> {
    if let Ok(s) = term.decode::<String>() {
        Ok(s)
    } else if let Ok(name) = term.atom_to_string() {
        Ok(name)
    } else if let Ok(i) = term.decode::<i64>() {
        Ok(i.to_string())
    } else {
        Err(rustler::Error::BadArg)
    }
}

// Decode a vector entry from an Elixir term (map, struct, or raw vector)
fn decode_vector_entry(term: Term) -> Result<VectorEntry, rustler::Error> {
    // If passed directly as a float list [1.0, 2.0, ...]
    if let Ok(vec) = term.decode::<Vec<f32>>() {
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

        for (k_term, v_term) in map_iter {
            let key: String = if let Ok(s) = k_term.decode::<String>() {
                s
            } else if let Ok(name) = k_term.atom_to_string() {
                name
            } else {
                continue;
            };

            match key.as_str() {
                "id" => {
                    if rustler::types::atom::nil() == v_term {
                        id_val = None;
                    } else if let Ok(s) = decode_id(v_term) {
                        id_val = Some(s);
                    } else {
                        return Err(rustler::Error::BadArg);
                    }
                }
                "vector" => {
                    if let Ok(vec) = v_term.decode::<Vec<f32>>() {
                        vector_val = Some(vec);
                    } else {
                        return Err(rustler::Error::BadArg);
                    }
                }
                "metadata" => {
                    if rustler::types::atom::nil() == v_term {
                        metadata_val = None;
                    } else if let Ok(serde_json::Value::Object(obj)) = term_to_json(v_term) {
                        metadata_val = Some(obj.into_iter().collect());
                    } else {
                        return Err(rustler::Error::BadArg);
                    }
                }
                _ => {}
            }
        }

        if let Some(vector) = vector_val {
            return Ok(VectorEntry {
                id: id_val,
                vector,
                metadata: metadata_val,
            });
        }
    }

    Err(rustler::Error::BadArg)
}

fn parse_metric_str(s: &str) -> DistanceMetric {
    match s.to_lowercase().as_str() {
        "euclidean" => DistanceMetric::Euclidean,
        "cosine" => DistanceMetric::Cosine,
        "dot_product" | "dotproduct" | "dot" => DistanceMetric::DotProduct,
        "manhattan" => DistanceMetric::Manhattan,
        _ => DistanceMetric::Cosine,
    }
}

fn parse_metric(term: Term) -> DistanceMetric {
    if let Ok(s) = term.atom_to_string() {
        parse_metric_str(&s)
    } else if let Ok(s) = term.decode::<String>() {
        parse_metric_str(&s)
    } else {
        DistanceMetric::Cosine
    }
}

fn parse_quantization(term: Term) -> Option<QuantizationConfig> {
    if rustler::types::atom::nil() == term {
        return None;
    }

    let q_str = if let Ok(s) = term.atom_to_string() {
        Some(s)
    } else if let Ok(s) = term.decode::<String>() {
        Some(s)
    } else {
        None
    };

    if let Some(s) = q_str {
        return match s.to_lowercase().as_str() {
            "scalar" => Some(QuantizationConfig::Scalar),
            "binary" => Some(QuantizationConfig::Binary),
            "none" => Some(QuantizationConfig::None),
            _ => None,
        };
    }

    if let Some(map_iter) = rustler::types::map::MapIterator::new(term) {
        let mut subspaces = 8usize;
        let mut k = 256usize;
        let mut is_product = false;

        for (k_term, v_term) in map_iter {
            let key: String = if let Ok(s) = k_term.decode::<String>() {
                s
            } else if let Ok(name) = k_term.atom_to_string() {
                name
            } else {
                continue;
            };

            match key.as_str() {
                "type" => {
                    if let Ok(s) = v_term.decode::<String>() {
                        if s.to_lowercase() == "scalar" {
                            return Some(QuantizationConfig::Scalar);
                        } else if s.to_lowercase() == "binary" {
                            return Some(QuantizationConfig::Binary);
                        } else if s.to_lowercase() == "product" {
                            is_product = true;
                        }
                    } else if let Ok(name) = v_term.atom_to_string() {
                        if name == "scalar" {
                            return Some(QuantizationConfig::Scalar);
                        } else if name == "binary" {
                            return Some(QuantizationConfig::Binary);
                        } else if name == "product" {
                            is_product = true;
                        }
                    }
                }
                "subspaces" => {
                    if let Ok(val) = v_term.decode::<usize>() {
                        subspaces = val;
                        is_product = true;
                    }
                }
                "k" => {
                    if let Ok(val) = v_term.decode::<usize>() {
                        k = val;
                        is_product = true;
                    }
                }
                _ => {}
            }
        }

        if is_product {
            return Some(QuantizationConfig::Product { subspaces, k });
        }
    }

    None
}

fn parse_db_options(path: String, dimensions: usize, opts_term: Option<Term>) -> DbOptions {
    let mut distance_metric = DistanceMetric::Cosine;
    let mut hnsw_config = None;
    let mut quantization = None;

    if let Some(opts) = opts_term {
        if let Some(map_iter) = rustler::types::map::MapIterator::new(opts) {
            for (k, v) in map_iter {
                let key: String = if let Ok(s) = k.decode::<String>() {
                    s
                } else if let Ok(name) = k.atom_to_string() {
                    name
                } else {
                    continue;
                };

                match key.as_str() {
                    "metric" | "distance_metric" => {
                        distance_metric = parse_metric(v);
                    }
                    "hnsw" => {
                        if let Ok(b) = v.decode::<bool>() {
                            if b {
                                hnsw_config = Some(HnswConfig::default());
                            }
                        } else if let Some(hnsw_map) = rustler::types::map::MapIterator::new(v) {
                            let mut cfg = HnswConfig::default();
                            for (hk, hv) in hnsw_map {
                                let hkey: String = if let Ok(s) = hk.decode::<String>() {
                                    s
                                } else if let Ok(name) = hk.atom_to_string() {
                                    name
                                } else {
                                    continue;
                                };
                                match hkey.as_str() {
                                    "m" => {
                                        if let Ok(val) = hv.decode::<usize>() {
                                            cfg.m = val;
                                        }
                                    }
                                    "ef_construction" => {
                                        if let Ok(val) = hv.decode::<usize>() {
                                            cfg.ef_construction = val;
                                        }
                                    }
                                    "ef_search" => {
                                        if let Ok(val) = hv.decode::<usize>() {
                                            cfg.ef_search = val;
                                        }
                                    }
                                    "max_elements" => {
                                        if let Ok(val) = hv.decode::<usize>() {
                                            cfg.max_elements = val;
                                        }
                                    }
                                    _ => {}
                                }
                            }
                            hnsw_config = Some(cfg);
                        }
                    }
                    "quantization" => {
                        quantization = parse_quantization(v);
                    }
                    _ => {}
                }
            }
        }
    }

    DbOptions {
        dimensions,
        distance_metric,
        storage_path: path,
        hnsw_config,
        quantization,
    }
}

fn open_db_impl<'a>(
    env: Env<'a>,
    path: String,
    dimensions: usize,
    options: Option<Term<'a>>,
) -> NifResult<Term<'a>> {
    let opts = parse_db_options(path, dimensions, options);

    match VectorDB::new(opts) {
        Ok(db) => {
            let resource = ResourceArc::new(DbResource { db });
            Ok((ok(), resource).encode(env))
        }
        Err(err) => Ok((error(), err.to_string()).encode(env)),
    }
}

#[rustler::nif(schedule = "DirtyIo")]
fn open_db<'a>(env: Env<'a>, path: String, dimensions: usize) -> NifResult<Term<'a>> {
    open_db_impl(env, path, dimensions, None)
}

#[rustler::nif(schedule = "DirtyIo")]
fn open_db_with_options<'a>(
    env: Env<'a>,
    path: String,
    dimensions: usize,
    options: Option<Term<'a>>,
) -> NifResult<Term<'a>> {
    open_db_impl(env, path, dimensions, options)
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

    match db_ref.db.insert(entry) {
        Ok(id) => Ok((ok(), id).encode(env)),
        Err(err) => Ok((error(), err.to_string()).encode(env)),
    }
}

#[rustler::nif(schedule = "DirtyIo")]
fn insert_batch<'a>(
    env: Env<'a>,
    db_ref: ResourceArc<DbResource>,
    entries_term: Vec<Term<'a>>,
) -> NifResult<Term<'a>> {
    let mut entries = Vec::with_capacity(entries_term.len());
    for t in entries_term {
        match decode_vector_entry(t) {
            Ok(e) => entries.push(e),
            Err(_) => return Ok((error(), "Invalid vector entry in batch").encode(env)),
        }
    }

    match db_ref.db.insert_batch(entries) {
        Ok(ids) => Ok((ok(), ids).encode(env)),
        Err(err) => Ok((error(), err.to_string()).encode(env)),
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

    match db_ref.db.get(&id_str) {
        Ok(Some(entry)) => {
            let mut map = Term::map_new(env);
            let entry_id = entry.id.as_deref().unwrap_or(&id_str);
            map = map.map_put(id().encode(env), entry_id.encode(env)).unwrap();
            map = map
                .map_put(vector().encode(env), entry.vector.encode(env))
                .unwrap();
            let meta_term = match entry.metadata {
                Some(m) => {
                    let json_obj = serde_json::Value::Object(m.into_iter().collect());
                    json_to_term(env, &json_obj)
                }
                None => rustler::types::atom::nil().encode(env),
            };
            map = map.map_put(metadata().encode(env), meta_term).unwrap();
            Ok((ok(), map).encode(env))
        }
        Ok(None) => Ok((error(), not_found()).encode(env)),
        Err(err) => Ok((error(), err.to_string()).encode(env)),
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

    match db_ref.db.delete(&id_str) {
        Ok(deleted) => Ok((ok(), deleted).encode(env)),
        Err(err) => Ok((error(), err.to_string()).encode(env)),
    }
}

#[rustler::nif]
fn db_len<'a>(env: Env<'a>, db_ref: ResourceArc<DbResource>) -> NifResult<Term<'a>> {
    match db_ref.db.len() {
        Ok(len) => Ok((ok(), len).encode(env)),
        Err(err) => Ok((error(), err.to_string()).encode(env)),
    }
}

#[rustler::nif]
fn is_empty<'a>(env: Env<'a>, db_ref: ResourceArc<DbResource>) -> NifResult<Term<'a>> {
    match db_ref.db.is_empty() {
        Ok(empty) => Ok((ok(), empty).encode(env)),
        Err(err) => Ok((error(), err.to_string()).encode(env)),
    }
}

#[rustler::nif(schedule = "DirtyIo")]
fn all_ids<'a>(env: Env<'a>, db_ref: ResourceArc<DbResource>) -> NifResult<Term<'a>> {
    match db_ref.db.keys() {
        Ok(ids) => Ok((ok(), ids).encode(env)),
        Err(err) => Ok((error(), err.to_string()).encode(env)),
    }
}

fn extract_filter<'a>(filter: Option<Term<'a>>) -> Option<HashMap<String, serde_json::Value>> {
    let f_term = filter?;
    if rustler::types::atom::nil() == f_term {
        return None;
    }
    if let Ok(serde_json::Value::Object(obj)) = term_to_json(f_term) {
        if obj.is_empty() {
            None
        } else {
            Some(obj.into_iter().collect())
        }
    } else if let Ok(list) = f_term.decode::<Vec<(Term, Term)>>() {
        let mut map = HashMap::new();
        for (k_term, v_term) in list {
            let key = if let Ok(s) = k_term.decode::<String>() {
                s
            } else if let Ok(name) = k_term.atom_to_string() {
                name
            } else {
                continue;
            };
            if let Ok(val) = term_to_json(v_term) {
                map.insert(key, val);
            }
        }
        if map.is_empty() {
            None
        } else {
            Some(map)
        }
    } else {
        None
    }
}

#[rustler::nif(schedule = "DirtyCpu")]
fn search<'a>(
    env: Env<'a>,
    db_ref: ResourceArc<DbResource>,
    query_vector: Vec<f32>,
    filter: Option<Term<'a>>,
    limit: usize,
) -> NifResult<Term<'a>> {
    let query_filter = extract_filter(filter);

    let query = SearchQuery {
        vector: query_vector,
        k: limit,
        filter: query_filter,
        ef_search: None,
    };

    match db_ref.db.search(query) {
        Ok(results) => {
            let ids: Vec<String> = results.into_iter().map(|r| r.id).collect();
            Ok(ids.encode(env))
        }
        Err(err) => Ok((error(), err.to_string()).encode(env)),
    }
}

#[rustler::nif(schedule = "DirtyCpu")]
fn search_detailed<'a>(
    env: Env<'a>,
    db_ref: ResourceArc<DbResource>,
    query_vector: Vec<f32>,
    filter: Option<Term<'a>>,
    limit: usize,
    ef_search: Option<usize>,
) -> NifResult<Term<'a>> {
    let query_filter = extract_filter(filter);

    let query = SearchQuery {
        vector: query_vector,
        k: limit,
        filter: query_filter,
        ef_search,
    };

    match db_ref.db.search(query) {
        Ok(results) => {
            let terms: Vec<Term<'a>> = results
                .into_iter()
                .map(|r| {
                    let mut map = Term::map_new(env);
                    map = map.map_put(id().encode(env), r.id.encode(env)).unwrap();
                    map = map
                        .map_put(score().encode(env), r.score.encode(env))
                        .unwrap();
                    let vec_term = match r.vector {
                        Some(v) => v.encode(env),
                        None => rustler::types::atom::nil().encode(env),
                    };
                    map = map.map_put(vector().encode(env), vec_term).unwrap();
                    let meta_term = match r.metadata {
                        Some(m) => {
                            let json_obj = serde_json::Value::Object(m.into_iter().collect());
                            json_to_term(env, &json_obj)
                        }
                        None => rustler::types::atom::nil().encode(env),
                    };
                    map = map.map_put(metadata().encode(env), meta_term).unwrap();
                    map
                })
                .collect();
            Ok((ok(), terms).encode(env))
        }
        Err(err) => Ok((error(), err.to_string()).encode(env)),
    }
}

#[rustler::nif]
fn calculate_distance<'a>(
    env: Env<'a>,
    a: Vec<f32>,
    b: Vec<f32>,
    metric_term: Term<'a>,
) -> NifResult<Term<'a>> {
    let metric = parse_metric(metric_term);
    match ruvector_core::distance::distance(&a, &b, metric) {
        Ok(dist) => Ok((ok(), dist).encode(env)),
        Err(err) => Ok((error(), err.to_string()).encode(env)),
    }
}

#[rustler::nif]
fn info<'a>(env: Env<'a>, db_ref: ResourceArc<DbResource>) -> NifResult<Term<'a>> {
    let len = db_ref.db.len().unwrap_or(0);
    let options = db_ref.db.options();

    let mut map = rustler::Term::map_new(env);
    map = map
        .map_put(
            rustler::Encoder::encode(&"node_count", env),
            rustler::Encoder::encode(&len, env),
        )
        .unwrap();
    map = map
        .map_put(
            rustler::Encoder::encode(&"dimensions", env),
            rustler::Encoder::encode(&options.dimensions, env),
        )
        .unwrap();
    let metric_str = match options.distance_metric {
        ruvector_core::types::DistanceMetric::Euclidean => "euclidean",
        ruvector_core::types::DistanceMetric::Cosine => "cosine",
        ruvector_core::types::DistanceMetric::DotProduct => "dot_product",
        ruvector_core::types::DistanceMetric::Manhattan => "manhattan",
    };
    map = map
        .map_put(
            rustler::Encoder::encode(&"distance_metric", env),
            rustler::Encoder::encode(&metric_str, env),
        )
        .unwrap();
    map = map
        .map_put(
            rustler::Encoder::encode(&"storage_path", env),
            rustler::Encoder::encode(&options.storage_path, env),
        )
        .unwrap();

    let hnsw_term = match &options.hnsw_config {
        Some(cfg) => {
            let mut hmap = rustler::Term::map_new(env);
            hmap = hmap
                .map_put(
                    rustler::Encoder::encode(&"m", env),
                    rustler::Encoder::encode(&cfg.m, env),
                )
                .unwrap();
            hmap = hmap
                .map_put(
                    rustler::Encoder::encode(&"ef_construction", env),
                    rustler::Encoder::encode(&cfg.ef_construction, env),
                )
                .unwrap();
            hmap = hmap
                .map_put(
                    rustler::Encoder::encode(&"ef_search", env),
                    rustler::Encoder::encode(&cfg.ef_search, env),
                )
                .unwrap();
            hmap = hmap
                .map_put(
                    rustler::Encoder::encode(&"max_elements", env),
                    rustler::Encoder::encode(&cfg.max_elements, env),
                )
                .unwrap();
            hmap
        }
        None => rustler::types::atom::nil().encode(env),
    };
    map = map
        .map_put(rustler::Encoder::encode(&"hnsw", env), hnsw_term)
        .unwrap();

    Ok(map)
}

rustler::init!("Elixir.RuvectorElixir", load = load);
