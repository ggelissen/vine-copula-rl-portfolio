# Data provenance and access

The manuscript analyses a seven-asset panel comprising three US price indices, three Chinese equity ETFs, and a Chinese gold ETF. The identifiers recorded in the manuscript are `.INX`, `.IXIC`, `.DJI`, `sh510050`, `sh510880`, `sz159915`, and `sz159934`; closing levels were obtained from Sina Finance through AKShare. The analysed common calendar covers 16 December 2013 to 6 July 2026 (2,946 closing levels and 2,945 daily log returns). The retrieval timestamp was not recorded and should not be inferred.

The study uses **price indices/ETF closing levels**, not a fully investable total-return panel. Dividend, FX, tax, and local-market execution conventions are therefore limitations. The last 24 scheduled monthly holding periods were reserved for evaluation; only 22 had complete realised returns in the reported comparison.

The three `portfolio_*.csv` files in this directory are candidate-universe inputs, not three interchangeable versions of the reported panel. Do not substitute another candidate file or refresh prices when attempting to reproduce a frozen result. A reproducible run must use the exact input hashes, asset ordering, split metadata, and transformation rules from its corresponding release manifest.

Source-data redistribution rights have not been established here. Before distributing raw or derived price files, check the terms of AKShare and the underlying provider. If redistribution is not permitted, obtain the series independently and compare hashes and calendar diagnostics; a fresh retrieval may not be byte-identical to the frozen dataset.

Generated synthetic bundles and checkpoints are experimental artifacts, not source data. Large files may require Git LFS or external archival storage. A successful `git clone` alone does not guarantee access to them.
