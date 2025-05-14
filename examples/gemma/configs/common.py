import dataclasses


@dataclasses.dataclass(unsafe_hash=True)
class MeshRules:
  embed: str | None = None
  mlp: str | None = None
  kv: str | None = None
  vocab: str | None = None

  def __call__(self, *keys: str) -> tuple[str, ...]:
    return tuple(
      getattr(self, key) if key is not None else None
      for key in keys
    )
