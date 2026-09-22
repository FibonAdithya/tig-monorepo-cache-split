def needs_index_build(batch) -> bool:
    """Whether this batch has a build phase.

    Membership, not truthiness. `build_fuel_budget == 0` is a real value -- "no
    build fuel granted" -- and `if batch.get("build_fuel_budget"):` would treat
    it as "this challenge has no build phase", silently skipping the build for a
    c004 batch and producing nonces that never load an index.

    As of this writing, no real batch produced by the master ever carries
    `build_fuel_budget`: wiring it into the master needs a postgres/init.sql
    schema migration, a Python reimplementation of the Rust fuel arithmetic
    (there is no FFI path from the Python master to the fuel-calculation
    crate), and edits to both `JSONB_BUILD_OBJECT` calls in slave_manager.py --
    all out of scope for this plan. So `needs_index_build` returning `False`
    for every live batch today is the expected state, not a regression; this
    function is the landing pad for when the master is wired.
    """
    return "build_fuel_budget" in batch
