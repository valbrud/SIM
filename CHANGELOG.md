# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

[1.0.0] - 2024-12-11

<!-- insertion marker -->
## [1.5.0](https://github.com/valbrud/SIM/releases/tag/1.5.0) - 2025-09-08

<small>[Compare with first commit](https://github.com/valbrud/SIM/compare/a558b5f90a910f6db23b274403aa1022bb1fe4de...1.5.0)</small>

### Added

- Added changed button functionality. Fixed GUI bugs ([6271577](https://github.com/valbrud/SIM/commit/6271577d6d90a7e7f527b117dcd047a73c582cf0) by valbrud).
- Added first complete impleemntation of the PatternEstimator class for estimation of the parameters of the illumination from data.  Two different implementations - through interpolation and cross-correlation are provided. Added modulation strength estimation as a part of the illumination class. ([8188571](https://github.com/valbrud/SIM/commit/8188571fde53e0dc3268ca034afebffcc08a9e34) by valbrud).
- Added new versioin of illumination classes, that supports initialization with experimental data (imperfect rotations and phase shifts) ([5c71c81](https://github.com/valbrud/SIM/commit/5c71c81befa193d14093f99c16092e94d2afd044) by valbrud).
- Added first (incomplete) version of the parameters estimate by interpolation ([00d2df7](https://github.com/valbrud/SIM/commit/00d2df767b68b9700891d6b08de96772ee674dd4) by valbrud).
- Added partial implementation of pattern estimation through interpolation ([7eea849](https://github.com/valbrud/SIM/commit/7eea8496949b72e68a9728f0192deb7b9d7a3ba6) by valbrud).
- Added (incomplete) parameter estimation. Added estimation of SSNR from data with binomial splitting. ([fc5d7aa](https://github.com/valbrud/SIM/commit/fc5d7aae06d6ff0716f455904985856688dba763) by valbrud).
- Added initial implementation of the ProcessorSIM class ([9ee83c8](https://github.com/valbrud/SIM/commit/9ee83c81766f74b9f0d04042f6ebabfce96ff7fb) by valbrud).
- Added automatic spatial shifts setter ([5b94ec5](https://github.com/valbrud/SIM/commit/5b94ec5e5d692e5244347116c54e39c50bbcd104) by valerii).
- Added non-linear SIM implementation ([8bc6822](https://github.com/valbrud/SIM/commit/8bc6822dfa6f3bf7c71c5fab9bfc8fde560d2c61) by valerii).
- Added functions for real and Fourier image deconvolution. ([c8ea21f](https://github.com/valbrud/SIM/commit/c8ea21f09ff29c8e00155711a737a2054c434192) by valerii).
- Added CHANGELOG.md, requirements.txt and other auxiliary files ([85fa0f0](https://github.com/valbrud/SIM/commit/85fa0f0cbd68f0dc490a5eaf192db1cbd4ff349d) by valerii).
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

- Fixed naming conventions ([0a03ef2](https://github.com/valbrud/SIM/commit/0a03ef2b345b572ac38081d606ad58a58854eda3) by valbrud).
- Fixed bugs, added better debugging pipeline ([9c3768a](https://github.com/valbrud/SIM/commit/9c3768aeb10aaa1276121bacac633319f5fcaf81) by valbrud).
- Fixed bugs in the reconstruction ([8371940](https://github.com/valbrud/SIM/commit/8371940b3f822e1df496e20843ea1df0381331b7) by valbrud).
- Fixed reconstruction issues ([bc70602](https://github.com/valbrud/SIM/commit/bc7060244b18712ef7d528d034aac69c6159c4b2) by valbrud).
- Fixed naming and normalization issues ([c877470](https://github.com/valbrud/SIM/commit/c877470cea65f72a0519061836806652bcfd4537) by valbrud).
- Fixed numerical bugs ([137a7f3](https://github.com/valbrud/SIM/commit/137a7f300fe91a19a6f00737b5b043dde9bbb20b) by valbrud).
- Fixed documentation. Fixed apodization. ([1efeab7](https://github.com/valbrud/SIM/commit/1efeab7b66ba1b066f3721e43d27e9195933a267) by valbrud).
- Fixed paths. Fixed reconstruction procedure. Added metaclass to ensure proper dimensionality checking. ([dfaa2b7](https://github.com/valbrud/SIM/commit/dfaa2b726720b2c33f3477ff3ec987182565abde) by valbrud).
- Fixed design imperfections ([a0e5d84](https://github.com/valbrud/SIM/commit/a0e5d84c05b873459a8d2ff8c6892e748d9b5dbc) by valerii).
- Fixed reconstruction algorithm. ([79acf21](https://github.com/valbrud/SIM/commit/79acf214dbbe734f0c9c209e0884b6c7761029d1) by valerii).
- Fixed numeric errors ([12e7c32](https://github.com/valbrud/SIM/commit/12e7c32c57af1347b6ca5555ae0bf1ffa0413b96) by valerii).
- Fixed code redundancy at the expense of slight generalization reduction. Fixed SIMulator class ([54fde9e](https://github.com/valbrud/SIM/commit/54fde9e26b3e311f783b1ffe113353f10233b0ce) by valerii).
- Fixed: Reconstruction has bugs remaining, but works for all cases. ([88de609](https://github.com/valbrud/SIM/commit/88de6090d7ceab890259f55ff470f2973017f3fd) by valerii).
- Fixed: Numerical peaks search is working ([56b34c1](https://github.com/valbrud/SIM/commit/56b34c126e348ba98b2352d5f4f4338e2b42e94c) by valerii).
- Fixed a mistake in Vj. SSNR functionality extended for lattice configuration ([4349ef2](https://github.com/valbrud/SIM/commit/4349ef2df7cab42e1f9fbe8c6cf1fd2a4b33bf95) by valerii).

### Changed

- Changed: Design of the PatternEstimator is more versitile and simple ([df793ec](https://github.com/valbrud/SIM/commit/df793ec9c76ed57264857c87db59e89e269b8d6e) by valbrud).
- Changed interface to the SSNR class, significantly reduced the number of classes ([e969835](https://github.com/valbrud/SIM/commit/e9698356913bc71760811e47e532a3b2034e7a4d) by valerii).
- Changed interface to the Illumination class, now it is the whole tree of classes ([43da4dd](https://github.com/valbrud/SIM/commit/43da4dd12f52aebfeba7ad47a1fb535e8029892b) by valerii).
- Changed: Article plots generating functions are moved to a separate folder. ([365a351](https://github.com/valbrud/SIM/commit/365a3511b3b98e54a2e857da1bba4d93fae2d32f) by valerii).
- Changed: Major renaming. Visualization extension. SSNR functionality improved. ([1930c94](https://github.com/valbrud/SIM/commit/1930c94f56e13d4a659e2a8c4bd04e7990d1083e) by valerii).
- Changed: common configurations now belong to a class. Intensity plane waves can now be computed from plane waves analytically ([bb6a74d](https://github.com/valbrud/SIM/commit/bb6a74d9ccc6f3590add084c921748524720bf35) by valerii).
- Changed: design rethinking ([bca05e3](https://github.com/valbrud/SIM/commit/bca05e3735611374ef650f773efe91b2caaa7543) by valerii).
- Changed: code was vectorized. ([406821e](https://github.com/valbrud/SIM/commit/406821ee1231be37074e7e35f99327a91db9ffe6) by valbrud).
