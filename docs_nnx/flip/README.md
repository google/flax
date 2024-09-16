# FLIP: Flax Improvement Process

Most changes can be discussed with simple issues/discussions and pull requests.

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

First, create an issue with the [FLIP label]. All pull requests that relate to
the FLIP (i.e. adding the FLIP itself as well as any implementing pull requests)
should be linked to this issue.

Then create a pull request that consists of a copy of the `0000-template.md`
renamed to `%04d-{short-title}.md` - with the number being the issue number.

[FLIP label]: https://github.com/google/flax/issues?q=label%3AFLIP
