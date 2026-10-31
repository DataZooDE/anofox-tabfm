# Troubleshooting

Every failure is a DuckDB exception that names the fix, e.g.:

```
Invalid Input Error: tabfm: model 'classification' is not downloaded.
Run: CALL tabfm_download('classification');
```

## Installing

* Without `-unsigned`, installing from `get.anofox.com` fails with *"Attempting
  to install an extension file that doesn't have a valid signature"*. The
  community repository needs no flag.
* `custom_extension_repository` applies to **every** install, including ones
  DuckDB triggers for you. If it is set before `httpfs` is present, DuckDB
  autoloads `httpfs` from `get.anofox.com`, which does not serve it, and the
  failure names `httpfs` rather than the setting. Install `httpfs` first.
