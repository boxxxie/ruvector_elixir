use rustler::{Encoder, Env, Error, NifResult, ResourceArc, Term};
use ruvector_core::types::{DbOptions, DistanceMetric, SearchQuery};
use ruvector_core::VectorDB;
use std::sync::RwLock;

// Wrap the VectorDB in a struct so we can implement the Resource trait
pub struct DbResource {
    pub db: RwLock<VectorDB>,
}

rustler::atoms! {
    ok,
    error,
    not_found,
}

#[allow(non_local_definitions)]
fn load(env: Env, _info: Term) -> bool {
    let _ = rustler::resource!(DbResource, env);
    true
}

#[rustler::nif(schedule = "DirtyIo")]
fn open_db(path: String, dimensions: usize) -> NifResult<(rustler::Atom, ResourceArc<DbResource>)> {
    let options = DbOptions {
        dimensions,
        distance_metric: DistanceMetric::Cosine,
        storage_path: path,
        hnsw_config: None, // Flat index / no hnsw config or default if needed
        quantization: None,
    };

    match VectorDB::new(options) {
        Ok(db) => {
            let resource = ResourceArc::new(DbResource {
                db: RwLock::new(db),
            });
            Ok((ok(), resource))
        }
        Err(err) => Err(Error::Term(Box::new(err.to_string()))),
    }
}

#[rustler::nif(schedule = "DirtyCpu")]
fn search(
    env: Env,
    db_ref: ResourceArc<DbResource>,
    query_vector: Vec<f32>,
    filter: Option<std::collections::HashMap<String, String>>,
    limit: usize,
) -> NifResult<Term> {
    let db = db_ref.db.read().unwrap();

    let query_filter: Option<std::collections::HashMap<String, serde_json::Value>> = match filter {
        Some(map) => {
            let mut filter_map = std::collections::HashMap::new();
            for (k, v) in map {
                filter_map.insert(k, serde_json::Value::String(v));
            }
            Some(filter_map)
        }
        None => None,
    };

    let query = SearchQuery {
        vector: query_vector,
        k: limit,
        filter: query_filter,
        ef_search: None,
    };

    match db.search(query) {
        Ok(results) => {
            let ids: Vec<String> = results.into_iter().map(|r| r.id).collect();
            Ok(ids.encode(env))
        }
        Err(err) => Err(Error::Term(Box::new(err.to_string()))),
    }
}

#[rustler::nif]
fn info(env: Env, db_ref: ResourceArc<DbResource>) -> NifResult<Term> {
    let db = db_ref.db.read().unwrap();
    let len = db.len().unwrap_or(0);

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
            rustler::Encoder::encode(&db.options().dimensions, env),
        )
        .unwrap();
    let metric_str = match db.options().distance_metric {
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

    Ok(map)
}

rustler::init!("Elixir.RuvectorElixir", load = load);
