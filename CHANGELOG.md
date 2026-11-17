# Changelog / release notes

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Possible sections in each release:

* Added: for new features.
* Changed: for changes in existing functionality.
* Deprecated: for soon-to-be removed features.
* Removed: for now removed features.
* Fixed: for any bug fixes.
* Security: in case of vulnerabilities.


## [v1.2.1](https://github.com/almarklein/simplezarr/releases/tag/v1.2.1) - 27-10-2026

### What's Changed
* Fix deka iso prefix by @almarklein in https://github.com/canpute/simplezarr/pull/58
* When writing a multi-pyramid file, the pyramid groups are ome-zarr too by @almarklein in https://github.com/canpute/simplezarr/pull/59

**Full Changelog**: https://github.com/canpute/simplezarr/compare/v1.2.0...v1.2.1


## [v1.2.0](https://github.com/almarklein/simplezarr/releases/tag/v1.2.0) - 23-10-2026

This release adds support for efficiently writing multiscale (ome-zarr) pyramids. It also comes with various fixes and a few speed improvements.

### What's Changed
* Support for building ome-zarr pyramids by @almarklein in https://github.com/canpute/simplezarr/pull/41
* Allow multiple multiscales in OME-Zarr pyramids by @almarklein in https://github.com/canpute/simplezarr/pull/45
* Fix how chunk_separator is handled by @almarklein in https://github.com/canpute/simplezarr/pull/46
* Fix chunk pool cache eviction by @almarklein in https://github.com/canpute/simplezarr/pull/47
* Fix some SI prefixes by @almarklein in https://github.com/canpute/simplezarr/pull/48
* Fix subtle index bounds check, plus test by @almarklein in https://github.com/canpute/simplezarr/pull/49
* Fix when multiple errors occue in threads by @almarklein in https://github.com/canpute/simplezarr/pull/50
* Fix that partial reads/writes cannot create a key by @almarklein in https://github.com/canpute/simplezarr/pull/51
* Make sure the ScaleInfo.level is consistent by @almarklein in https://github.com/canpute/simplezarr/pull/52
* Fix that .attributes was None instead of {} by @almarklein in https://github.com/canpute/simplezarr/pull/53
* Fix various smaller issues by @almarklein in https://github.com/canpute/simplezarr/pull/54
* Performance improvements by @almarklein in https://github.com/canpute/simplezarr/pull/56
* Bump version by @almarklein in https://github.com/canpute/simplezarr/pull/57

**Full Changelog**: https://github.com/canpute/simplezarr/compare/v1.1.0...v1.2.0


## [v1.1.0](https://github.com/almarklein/simplezarr/releases/tag/v1.1.0) - 24-07-2026

### What's Changed
* Improve code and docs for creating custom codecs by @almarklein in https://github.com/canpute/simplezarr/pull/43

**Full Changelog**: https://github.com/canpute/simplezarr/compare/v1.0.0...v1.1.0


## [v1.0.0](https://github.com/almarklein/simplezarr/releases/tag/v1.0.0) - 11-06-2026

Second version. Now with proper support for indexing, creating arrays, and more.

### What's Changed
* Implement indexing by @almarklein in https://github.com/canpute/simplezarr/pull/20
* Improve docs and cleaner stores by @almarklein in https://github.com/canpute/simplezarr/pull/25
* Improve stores some more by @almarklein in https://github.com/canpute/simplezarr/pull/26
* Implement ZarrrGroup.create() and ZarrArray.create() by @almarklein in https://github.com/canpute/simplezarr/pull/27
* Improvements to nodes module testing by @almarklein in https://github.com/canpute/simplezarr/pull/28
* Add benchmark script by @almarklein in https://github.com/canpute/simplezarr/pull/29
* Tweak readme and docs by @almarklein in https://github.com/canpute/simplezarr/pull/31
* Mark extension points in the code by @almarklein in https://github.com/canpute/simplezarr/pull/32
* Addendum to #32 by @almarklein in https://github.com/canpute/simplezarr/pull/33
* Add path arg to open_zarr() by @almarklein in https://github.com/canpute/simplezarr/pull/35
* Explicitly adopt the convention that the top-left pixel center represent the origin in data-space by @almarklein in https://github.com/canpute/simplezarr/pull/36
* get_chunk and set_chunk don't allow index outside of chunkgrid by @almarklein in https://github.com/canpute/simplezarr/pull/37
* Prevent hanging when getting/setting empty slice by @almarklein in https://github.com/canpute/simplezarr/pull/39
* Transform to pixel-center convention when missing translations by @almarklein in https://github.com/canpute/simplezarr/pull/40
* Bump version by @almarklein in https://github.com/canpute/simplezarr/pull/42

**Full Changelog**: https://github.com/canpute/simplezarr/compare/v0.0.1...v1.0.0


## [v0.0.1](https://github.com/almarklein/simplezarr/releases/tag/v0.0.1) - 11-06-2026

First version.

### What's Changed

* First version that can navigate a file somewhat by @almarklein in https://github.com/canpute/simplezarr/pull/1
* flake etc by @almarklein in https://github.com/canpute/simplezarr/pull/2
* Store tests by @almarklein in https://github.com/canpute/simplezarr/pull/3
* Add codecs for reading and writing chunks by @almarklein in https://github.com/canpute/simplezarr/pull/4
* Implement nodes properly by @almarklein in https://github.com/canpute/simplezarr/pull/8
* add slow store by @clewis7 in https://github.com/canpute/simplezarr/pull/11
* Add support for multiscale (i.e. OME-Zarr) via utils by @almarklein in https://github.com/canpute/simplezarr/pull/10
* Misc by @almarklein in https://github.com/canpute/simplezarr/pull/14
* Fix LocalStore for Windows by @almarklein in https://github.com/canpute/simplezarr/pull/16
* Add CI by @almarklein in https://github.com/canpute/simplezarr/pull/15
* Docs by @almarklein in https://github.com/canpute/simplezarr/pull/18

### New Contributors
* @clewis7 made their first contribution in https://github.com/canpute/simplezarr/pull/11

**Full Changelog**: https://github.com/canpute/simplezarr/commits/v0.0.1
