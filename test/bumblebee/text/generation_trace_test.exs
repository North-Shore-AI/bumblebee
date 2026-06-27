defmodule Bumblebee.Text.GenerationTraceTest do
  use ExUnit.Case, async: true

  test "greedy generation can return per-step logits and cache offsets" do
    spec =
      Bumblebee.configure(Bumblebee.Text.Gpt2,
        architecture: :for_causal_language_modeling,
        vocab_size: 32,
        hidden_size: 4,
        num_blocks: 2,
        num_attention_heads: 2,
        max_positions: 8,
        intermediate_size: 8,
        dropout_rate: 0.0,
        embeddings_dropout_rate: 0.0,
        attention_dropout_rate: 0.0
      )

    model = Bumblebee.build_model(spec)
    {init_fn, _predict_fn} = Axon.build(model)

    inputs = %{
      "input_ids" => Nx.tensor([[1, 2, 3]], type: :u32),
      "seed" => Nx.tensor([0])
    }

    params = init_fn.(%{"input_ids" => Nx.template({1, 3}, :u32)}, Axon.ModelState.empty())

    generation_config =
      Bumblebee.configure(Bumblebee.Text.GenerationConfig,
        max_new_tokens: 2,
        pad_token_id: 0,
        eos_token_id: nil
      )

    generate = Bumblebee.Text.Generation.build_generate(model, spec, generation_config)

    traced_generate =
      Bumblebee.Text.Generation.build_generate(model, spec, generation_config, trace: true)

    plain = generate.(params, inputs)
    traced = traced_generate.(params, inputs)

    assert Nx.to_flat_list(traced.token_ids) == Nx.to_flat_list(plain.token_ids)
    assert Nx.shape(traced.trace_logits) == {1, 2, 32}
    assert Nx.shape(traced.trace_cache_offsets) == {1, 2}
    assert Nx.to_flat_list(traced.trace_cache_offsets) == [3, 4]
  end
end
