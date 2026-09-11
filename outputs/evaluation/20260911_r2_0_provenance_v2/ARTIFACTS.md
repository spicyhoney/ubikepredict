# Artifact delivery

Small reports, CSV files, figures, commands, logs and manifests may be committed. Row-level parquet files are reproducible from the recorded command and exact input hashes; large derivatives may remain outside Git when repository size is a concern. The manifest records every artifact's byte size and SHA256.
