# Design Note: transformation lifting

## Introduction

JAX uses a functional api meaning that it deals only which pure functions only.
A pure function is defined as a function where the output only depends on the function arguments.
Therefore, mutable state outside the function should not affect the function itself and the function
itself should not cause side effects in objects that live outside of the function.

Python functions do not have to be pure because they allow side effects or mutations to occur.
For JAX restricting the API to pure functions has a number of advantages:

1. It becames easier to reason about functions locally
2. Both stochasticity and determinism are explicit because a function can only return a different output if the arguments are changed.
3. Functional transforms which would otherwise be ambigious. 

## Functionalization

TODO

## Lifting

TODO

## Alternatives

TODO