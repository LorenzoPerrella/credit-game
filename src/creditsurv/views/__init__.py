"""The tables behind the documentation site's views.

A view is an aggregate: a table small enough to commit, computed locally from the cells, the
cached fit and the ingested parquet, because the data behind it cannot reach the CI that
publishes the site. The site turns each table into an interactive figure at build time.

* :mod:`creditsurv.views.segments` -- the sub-items a view can be opened by, defined once;
* :mod:`creditsurv.views.calibration` -- non-parametric against parametric, by any segment;
* :mod:`creditsurv.views.tables` -- the committed format, a parquet per view and a manifest.
"""
