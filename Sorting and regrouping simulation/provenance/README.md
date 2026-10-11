# Numerical fingerprint

`final_numeric_fingerprint.json` records SHA-256 fingerprints of the supplied result CSV content. Gzip container timestamps are excluded from these fingerprints.

`../verify_final.py` uses this file to compare simulation tables with the supplied numerical reference. Verification also checks input and draw hashes, metric identities, current source files and Figure 7 sources/exports. New full runs bind source hashes to their own run manifest and include a generated results report.
