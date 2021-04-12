# The Flax Philosophy

(in no particular order)

* Library code should be easy to read and understand.

* Prefer duplicating code over a bad abstraction.

* Generally, prefer duplicating code over adding options to functions.

* Comment-driven design: If it's hard to document your code, consider
  changing the design.

* Unit test-driven design: If it's hard to test your code, consider
  changing the design.

* People start projects by copying an existing implementation -- make
  base implementations excellent.

* If we expose an abstraction to our developers, we own the mental
  overhead.

* Developer-facing functional programming abstractions confuse some users,
  expose them where the benefit is high.

* "Read the manual" is not an appropriate response to developer confusion.
  The framework should guide developers
  towards good solutions, e.g. through assertions and error messages.

* An unhelpful error message is a bug.

* "Debugging is twice as hard as writing the code in the first
  place. Therefore, if you write the code as cleverly as possible, you
  are, by definition, not smart enough to debug it." -Brian Kernighan



