# FLIP : Flax Improvement Process

Most changes can be discussed in issues/discussions and with pull requests.

Some changes though are a bit larger in scope or require more discussion, and
these should be implemented as FLIPs. This allows for writing longer documents
that can be discussed in a pull request themselves.

The structure of FLIPs is kept as lightweight as possible to start and might
be extended later on.

## When you should use a FLIP

- When your change requires a design doc. We prefer collecting the designs as
  FLIPs for better discoverability and further reference.

- When your change requires extensive discussion. It's fine to have relatively
  short discussions on issues or pull requests, but when the discussion gets
  longer this becomes unpractical for later digestion. FLIPs allow to update the
  main document with a summary of the discussion and these updates can be
  discussed themselves in the pull request adding the FLIP.

## How to start a FLIP

First, create an issue with the [FLIP label].

Make a copy of `0000-template.md` and rename it to `0000-your-flip.md` - the
number will later be changed to the pull request that dded the FLIP.

[FLIP label]: https://github.com/google/flax/issues?q=label%3A%22Type%3A+FLIP%22