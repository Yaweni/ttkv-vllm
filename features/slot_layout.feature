Feature: The cache layout is derived from geometry, never carried across it

  TT-KV's two lane knobs are not the same kind of quantity, and that is the whole reason this
  feature exists. `res_k` comes from `res_frac`, a FRACTION of the head dim, so it follows the
  geometry on its own. `k_lanes` (the layout's `v_lanes`, and the K residual's lane count) is an
  ABSOLUTE count tuned at d=128, so it does not: at d=64 the shipped 81 lanes do not exist, and at
  d=256 the same 81 lanes are a third of the row rather than two thirds -- which reads as a HIGHER
  ratio while being a coarser approximation. A layout that accepted either case silently would let a
  config flatter itself.

  Scenario: One token's slot is billed the way the container bills it
    Given the shipped config "safe"
    When I account for one token's K slot
    Then the slot bytes match the container's own row bytes

  Scenario: A config with an absolute lane count is refused at a narrower head
    Given the shipped config "safe" with its absolute 81 lanes
    When I build a layout for head dim 64
    Then the layout refuses it

  Scenario: The ratio barely depends on how the head dim is split
    Given a fixed KV width and a fixed lane fraction
    When I build layouts for head dims 64, 128, 256 and 512
    Then every joint ratio stays between 4.2x and 4.6x
