project(
  'flaxlib',
  'cpp',
  version: '0.0.1',
  default_options: ['cpp_std=c++17'],
)
py = import('python').find_installation()
nanobind_dep = dependency('nanobind', static: true)
py.extension_module(
  'flaxlib',
  sources: ['src/lib.cc'],
  dependencies: [nanobind_dep],
  install: true,
)