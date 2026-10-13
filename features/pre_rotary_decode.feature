Feature: A pre-rotary store decodes to the same scores as materialise-then-rotate

  The store holds POSITION-FREE keys. Absorption -- the trick that makes the post-rotary path cheap,
  folding the decoder into the query -- is unavailable, because the rotation is a function of the
  KEY's position and so sits between the reconstruction and the contraction. Every scenario here
  exists because getting it wrong returns numbers that are finite, correctly shaped and plausible,
  so nothing downstream can catch it.

  Scenario: The Triton kernel agrees with the reference decode
    Given a packed K store for the shipped config "campaign4x"
    And post-rotary queries and per-row angle tables
    When the Triton kernel decodes the store
    Then its scores agree with the reference to 1e-4 relative

  Scenario: Rotating a stored row by its neighbour's position moves the scores
    Given a packed K store for the shipped config "campaign4x"
    And post-rotary queries and per-row angle tables
    When the kernel decodes with the angle table rolled by one position
    Then the scores move by more than 1e-2 relative

  Scenario: Omitting the rotation entirely moves the scores a lot
    Given a packed K store for the shipped config "campaign4x"
    And post-rotary queries and per-row angle tables
    When the kernel decodes with an identity rotation
    Then the scores move by more than 1e-2 relative

  Scenario: A position-free store is refused where the correct path is unavailable
    Given a layer whose storage mode is "pre"
    When the direct-contract path is asked to decode it
    Then it raises rather than returning numbers
