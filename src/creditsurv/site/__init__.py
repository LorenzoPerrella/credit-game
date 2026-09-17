"""The documentation site: figures, tables and numbers built from the committed views.

The views are computed locally by ``creditsurv views`` and committed under ``docs/tables``;
this package turns them into what a page shows, and :mod:`creditsurv.site.hooks` puts that
into the pages when MkDocs builds them. Nothing here reads the loan data, so the site builds
in CI, where the data cannot go.
"""
