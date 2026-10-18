//! Test support: one in-memory database connected for the whole test process.

use once_cell::sync::Lazy;

static DB_RUNTIME: Lazy<tokio::runtime::Runtime> = Lazy::new(|| {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("test runtime")
});

static CONNECTED: Lazy<()> = Lazy::new(|| {
    std::thread::scope(|scope| {
        scope
            .spawn(|| {
                DB_RUNTIME.block_on(async {
                    use surrealdb::engine::local::Mem;
                    crate::database::DB.connect::<Mem>(()).await.expect("connect the in-memory database");
                    crate::database::DB
                        .use_ns(crate::database::NS)
                        .use_db(crate::database::DB_NAME)
                        .await
                        .expect("select the namespace and database");
                });
            })
            .join()
            .expect("connect thread");
    });
});

/// Connects the global database to memory once; later calls are free.
pub(crate) fn mem_db() {
    Lazy::force(&CONNECTED);
}
