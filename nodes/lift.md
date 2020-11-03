# Design Note: transformation lifting

# Introduction

JAX uses a functional api meaning that it deals only which pure functions only.
Python functions do not have to be pure because they allow side effects or mutations to occur.

