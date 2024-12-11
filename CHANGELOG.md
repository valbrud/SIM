# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

[1.0.0] - 2024-12-11

<!-- insertion marker -->
## Unreleased

<small>[Compare with latest](https://github.com/valbrud/SIM/compare/a558b5f90a910f6db23b274403aa1022bb1fe4de...HEAD)</small>

### Added

- Added first version of the documentation ([edfe187](https://github.com/valbrud/SIM/commit/edfe187c6ce45b7f2a43c17fa05e9e4d7bb5b2b3) by valerii).
- Added random lines generator. 2D local reconstructions verified ([7e478ef](https://github.com/valbrud/SIM/commit/7e478ef6b6404e6b0bf62df7308a071c60341058) by valerii).
- Added 2D SSNR calculator ([b2b03f6](https://github.com/valbrud/SIM/commit/b2b03f6f79a0d7bcb0ec5a9dadf64e4331658d0c) by valerii).
- Added radially symmetric kernels. ([f8756e9](https://github.com/valbrud/SIM/commit/f8756e95478eb2f61e4bebbda98529fd40076a5c) by valerii).
- Added Finite kernel SDR class. Tests updated. Minor bugs fixed ([7976db3](https://github.com/valbrud/SIM/commit/7976db3c4b3b5d77ca05fc16e5b39a4f7b0e5c13) by valerii).
- Added Wiener and Flat noise filters and corresponding tests ([e9ad75d](https://github.com/valbrud/SIM/commit/e9ad75d874273c3db843bfd39b2c81c636002460) by valerii).
- Added two modalities of 3d SIM ([35a9f5f](https://github.com/valbrud/SIM/commit/35a9f5f638013092c60c43314c910f7bd2b993f9) by valerii).
- Added SIM shifts ([f6e49c3](https://github.com/valbrud/SIM/commit/f6e49c3d1c8d3dda57870389d0d46b1b4bd79982) by valerii).
- Added a short script for computation of truly periodic 3d lattices. The code for Vj is being reimplemented (again) ([d10a6a9](https://github.com/valbrud/SIM/commit/d10a6a9d655bf9c66f74d5804b3ae68cf95415a7) by valerii).
- Added Surface levels animations ([0a3c22e](https://github.com/valbrud/SIM/commit/0a3c22ec59c2e3b7641b33e494a98f8036d4f951) by valerii).
- Added Winodwing ([2ba592f](https://github.com/valbrud/SIM/commit/2ba592f2f0cc1cae49fad63aef78ab84b55829cd) by valerii).
- Added ring averaging. Tests extended ([a3a1689](https://github.com/valbrud/SIM/commit/a3a1689beefd1d6193ef6a8fe5266d0c0bc3716f) by valerii).
- Added Fourier based interpolation. ([e49166b](https://github.com/valbrud/SIM/commit/e49166b43848ef9977d6fc279725f8fd5bdd06d1) by valerii).
- Added SSNR calculator. ([933a197](https://github.com/valbrud/SIM/commit/933a1977b9997801d0f55875d8be0a6aec43fa7a) by valerii).
- Added GUI, parser, configuration files. Fixed errors ([2d25069](https://github.com/valbrud/SIM/commit/2d2506931215701cd81f878e21eeb8756310541c) by valerii).
- Added PlaneWave, VectorOperation and Box classes. Computation of an interference pattern of an arbitrary number of plain waves is possible ([a589e27](https://github.com/valbrud/SIM/commit/a589e279a2c41099fc9e04721fe61842a306d327) by valbrud).

### Fixed

- Fixed: Reconstruction has bugs remaining, but works for all cases. ([88de609](https://github.com/valbrud/SIM/commit/88de6090d7ceab890259f55ff470f2973017f3fd) by valerii).
- Fixed: Numerical peaks search is working ([56b34c1](https://github.com/valbrud/SIM/commit/56b34c126e348ba98b2352d5f4f4338e2b42e94c) by valerii).
- Fixed a mistake in Vj. SSNR functionality extended for lattice configuration ([4349ef2](https://github.com/valbrud/SIM/commit/4349ef2df7cab42e1f9fbe8c6cf1fd2a4b33bf95) by valerii).

### Changed

- Changed: Article plots generating functions are moved to a separate folder. ([365a351](https://github.com/valbrud/SIM/commit/365a3511b3b98e54a2e857da1bba4d93fae2d32f) by valerii).
- Changed: Major renaming. Visualization extension. SSNR functionality improved. ([1930c94](https://github.com/valbrud/SIM/commit/1930c94f56e13d4a659e2a8c4bd04e7990d1083e) by valerii).
- Changed: common configurations now belong to a class. Intensity plane waves can now be computed from plane waves analytically ([bb6a74d](https://github.com/valbrud/SIM/commit/bb6a74d9ccc6f3590add084c921748524720bf35) by valerii).
- Changed: design rethinking ([bca05e3](https://github.com/valbrud/SIM/commit/bca05e3735611374ef650f773efe91b2caaa7543) by valerii).
- Changed: code was vectorized. ([406821e](https://github.com/valbrud/SIM/commit/406821ee1231be37074e7e35f99327a91db9ffe6) by valbrud).

<!-- insertion marker -->
