Feature: A real model forward through the cache reproduces dense

  This layer compares against neither a synthetic store nor a torch oracle. It drives an actual
  transformers model through `make_cache`, which is the only way to exercise the
  `Cache`/`DynamicLayer` contract, the dtype and shape the model hands `update`, the
  un-rotate/re-rotate round trip inside a live attention, and whether the logits survive any of it.

  Scenario: A nearly lossless store tracks dense under both storage modes
    Given a real model and a nearly lossless config
    When I prefill through the cache with storage "post"
    Then the logits track dense to cosine 0.9999
    When I prefill through the cache with storage "pre"
    Then the logits track dense to cosine 0.9999

  Scenario: The pre-rotary store is the smaller one at a shippable config
    Given a real model and a shippable config
    When I prefill through the cache under both storage modes
    Then both stores are smaller than the dense cache
    And the pre-rotary store is smaller than the post-rotary one

  Scenario: Greedy decode matches dense through a folded store
    Given a real model and a nearly lossless config
    And a fold interval of 2 tokens so the tail folds during decode
    When I prefill and then decode four tokens greedily
    Then the greedy tokens match the dense ones exactly
    And the store actually folded
