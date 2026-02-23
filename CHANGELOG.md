# Changelog

## [0.0.17](https://github.com/adhityaravi/theow/compare/theow-v0.0.16...theow-v0.0.17) (2026-02-23)


### Features

* adds multi-model routing using the secondary llm ([63d0096](https://github.com/adhityaravi/theow/commit/63d00968f62236b8007e53a14984885057224eb1))

## [0.0.16](https://github.com/adhityaravi/theow/compare/theow-v0.0.15...theow-v0.0.16) (2026-02-23)


### Features

* chained recovery logic to attempt recovery on multiple depths ([358e6a1](https://github.com/adhityaravi/theow/commit/358e6a1480fdabb105f8f7036abb82a48b16f9e9))

## [0.0.15](https://github.com/adhityaravi/theow/compare/theow-v0.0.14...theow-v0.0.15) (2026-02-22)


### Miscellaneous

* update observation schema ([59c22a8](https://github.com/adhityaravi/theow/commit/59c22a80f10d136023d6cec0931e39fef98f95d1))

## [0.0.14](https://github.com/adhityaravi/theow/compare/theow-v0.0.13...theow-v0.0.14) (2026-02-22)


### Bug Fixes

* fixes give up calls ([ad74f9c](https://github.com/adhityaravi/theow/commit/ad74f9c39430632d476e8cd482d50fdbb321a461))

## [0.0.13](https://github.com/adhityaravi/theow/compare/theow-v0.0.12...theow-v0.0.13) (2026-02-22)


### Bug Fixes

* fixes tool building for openai like api ([3a230ba](https://github.com/adhityaravi/theow/commit/3a230bae3605dc704ac1d43a0bc98d278a7ee448))

## [0.0.12](https://github.com/adhityaravi/theow/compare/theow-v0.0.11...theow-v0.0.12) (2026-02-22)


### Bug Fixes

* fix hint passthrough ([168e9b6](https://github.com/adhityaravi/theow/commit/168e9b6e8895e1d0853239c999747e9bb43daa55))

## [0.0.11](https://github.com/adhityaravi/theow/compare/theow-v0.0.10...theow-v0.0.11) (2026-02-22)


### Features

* adds a dec specific hint ([37582f2](https://github.com/adhityaravi/theow/commit/37582f2b5d003b68a607e50a6c2e645cc9ffd350))

## [0.0.10](https://github.com/adhityaravi/theow/compare/theow-v0.0.9...theow-v0.0.10) (2026-02-22)


### Features

* improves vector search quality ([e6bb1cf](https://github.com/adhityaravi/theow/commit/e6bb1cff40f950b74bfee8451923e81e7ec39ed7))

## [0.0.9](https://github.com/adhityaravi/theow/compare/theow-v0.0.8...theow-v0.0.9) (2026-02-21)


### Features

* improve theows memory ops ([61b6e74](https://github.com/adhityaravi/theow/commit/61b6e74ed770c2d455210f12435c0c10068b48be))

## [0.0.8](https://github.com/adhityaravi/theow/compare/theow-v0.0.7...theow-v0.0.8) (2026-02-20)


### Bug Fixes

* enforce token limit on anthropic and gemini ([07eb0a7](https://github.com/adhityaravi/theow/commit/07eb0a73e10f52a2fb3f69980786ffea4caf767b))

## [0.0.7](https://github.com/adhityaravi/theow/compare/theow-v0.0.6...theow-v0.0.7) (2026-02-19)


### Features

* cli mode ([#6](https://github.com/adhityaravi/theow/issues/6)) ([58baa4f](https://github.com/adhityaravi/theow/commit/58baa4ffffdb70efa159b5770452a49633131093))

## [0.0.6](https://github.com/adhityaravi/theow/compare/theow-v0.0.5...theow-v0.0.6) (2026-02-17)


### Bug Fixes

* adds py.typed ([a49ec78](https://github.com/adhityaravi/theow/commit/a49ec789b75dfa133e5f6cca29b5703d09466a9f))

## [0.0.5](https://github.com/adhityaravi/theow/compare/theow-v0.0.4...theow-v0.0.5) (2026-02-17)


### Bug Fixes

* type hint issues ([63f4cfd](https://github.com/adhityaravi/theow/commit/63f4cfdcf8421d54280f7da0103e3fa5057d90e0))

## [0.0.4](https://github.com/adhityaravi/theow/compare/theow-v0.0.3...theow-v0.0.4) (2026-02-17)


### Features

* adds lifecycle hooks for decorator ([ed88335](https://github.com/adhityaravi/theow/commit/ed88335eeab3185ca5a30ce382777c3f2376dc35))


### Bug Fixes

* fix case specific prompts ([bdda658](https://github.com/adhityaravi/theow/commit/bdda658c01ca84754e7bd81eba629a7d9dad167c))
* force capping on tokens and tool calls ([8c7b128](https://github.com/adhityaravi/theow/commit/8c7b128dc94ea7f5d9b4311e43e6521a46c58b29))

## [0.0.3](https://github.com/adhityaravi/theow/compare/theow-v0.0.2...theow-v0.0.3) (2026-02-16)


### Features

* try matched rules in the order of match accuracy ([b2e808c](https://github.com/adhityaravi/theow/commit/b2e808c26e77891817f5adc9fec5dbc2b047ae4b))

## [0.0.2](https://github.com/adhityaravi/theow/compare/theow-v0.0.1...theow-v0.0.2) (2026-02-12)


### Features

* add explorer and tests ([bb3f767](https://github.com/adhityaravi/theow/commit/bb3f7673eaab2f51b60e2532b965e28d6a313130))
* add init files & update spec ([56c4646](https://github.com/adhityaravi/theow/commit/56c4646f18944c17ccd5852595954570ff9d1641))
* add initial resolver and helpers for theow ([78174f5](https://github.com/adhityaravi/theow/commit/78174f55d4503cee42211a5f5234c2a58d1fb2ff))
* adds a copilot gateway with minor refactors ([8dcb1aa](https://github.com/adhityaravi/theow/commit/8dcb1aa30ab09df07d6d04c1b0a2dd9269e21da5))
* adds anthropic and gemini gateway ([9009f4a](https://github.com/adhityaravi/theow/commit/9009f4a9714445b1e2ef3b82ffb461a2ec799743))
* adds theow built-in tools ([b52ddca](https://github.com/adhityaravi/theow/commit/b52ddca11d88adda429f6de4a70c6df786553139))
* adds theow explorer and engine ([55545c4](https://github.com/adhityaravi/theow/commit/55545c4208e51b7531c13ea7e60dc14a10ca307a))
* **spec:** add initial specs for sd-tools enhancement ([82dcc96](https://github.com/adhityaravi/theow/commit/82dcc963b4989555b7282a163981269640a6aac9))
