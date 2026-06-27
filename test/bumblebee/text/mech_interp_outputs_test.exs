defmodule Bumblebee.Text.MechInterpOutputsTest do
  use ExUnit.Case, async: true

  describe "GPT-2 deep transformer outputs" do
    test "remain disabled unless their global layer options are set" do
      outputs = run_gpt2([])

      assert %Axon.None{} = outputs.attention_queries
      assert %Axon.None{} = outputs.attention_scores
      assert %Axon.None{} = outputs.mlp_pre_activations
      assert %Axon.None{} = outputs.residual_streams_pre
      assert %Axon.None{} = outputs.norm_scales
      assert %Axon.None{} = outputs.norm_normalized
    end

    test "include attention, MLP, residual, and norm activations when opted in" do
      outputs =
        run_gpt2(
          output_attention_qkv: true,
          output_attention_scores: true,
          output_mlp_activations: true,
          output_residual_streams: true,
          output_norm_telemetry: true
        )

      assert tuple_size(outputs.attention_queries) == 2
      assert Nx.shape(elem(outputs.attention_queries, 0)) == {1, 3, 2, 2}
      assert Nx.shape(elem(outputs.attention_keys, 0)) == {1, 3, 2, 2}
      assert Nx.shape(elem(outputs.attention_values, 0)) == {1, 3, 2, 2}
      assert Nx.shape(elem(outputs.attention_scores, 0)) == {1, 2, 3, 3}
      assert Nx.shape(elem(outputs.attention_zs, 0)) == {1, 3, 2, 2}
      assert Nx.shape(elem(outputs.attention_outputs, 0)) == {1, 3, 4}

      assert tuple_size(outputs.mlp_pre_activations) == 2
      assert Nx.shape(elem(outputs.mlp_inputs, 0)) == {1, 3, 4}
      assert Nx.shape(elem(outputs.mlp_pre_activations, 0)) == {1, 3, 8}
      assert Nx.shape(elem(outputs.mlp_post_activations, 0)) == {1, 3, 8}
      assert Nx.shape(elem(outputs.mlp_outputs, 0)) == {1, 3, 4}

      assert tuple_size(outputs.residual_streams_pre) == 2
      assert Nx.shape(elem(outputs.residual_streams_pre, 0)) == {1, 3, 4}
      assert Nx.shape(elem(outputs.residual_streams_mid, 0)) == {1, 3, 4}
      assert Nx.shape(elem(outputs.residual_streams_post, 0)) == {1, 3, 4}

      assert Nx.shape(outputs.norm_scales) == {1, 3, 1}
      assert Nx.shape(outputs.norm_normalized) == {1, 3, 4}
    end

    test "causal LM heads preserve final norm telemetry when opted in" do
      outputs =
        run_gpt2(
          :for_causal_language_modeling,
          output_norm_telemetry: true
        )

      assert Nx.shape(outputs.logits) == {1, 3, 32}
      assert Nx.shape(outputs.norm_scales) == {1, 3, 1}
      assert Nx.shape(outputs.norm_normalized) == {1, 3, 4}
    end
  end

  describe "Qwen3 gated MLP outputs" do
    test "include real gated feed-forward activations when opted in" do
      outputs =
        run_qwen3(
          output_attention_qkv: true,
          output_attention_scores: true,
          output_mlp_activations: true,
          output_norm_telemetry: true
        )

      assert Nx.shape(elem(outputs.attention_queries, 0)) == {1, 3, 2, 2}
      assert Nx.shape(elem(outputs.attention_keys, 0)) == {1, 3, 1, 2}
      assert Nx.shape(elem(outputs.attention_values, 0)) == {1, 3, 1, 2}
      assert Nx.shape(elem(outputs.attention_scores, 0)) == {1, 2, 3, 3}

      assert tuple_size(outputs.mlp_pre_activations) == 2
      assert Nx.shape(elem(outputs.mlp_inputs, 0)) == {1, 3, 4}
      assert Nx.shape(elem(outputs.mlp_pre_activations, 0)) == {1, 3, 8}
      assert Nx.shape(elem(outputs.mlp_post_activations, 0)) == {1, 3, 8}
      assert Nx.shape(elem(outputs.mlp_outputs, 0)) == {1, 3, 4}

      assert Nx.shape(outputs.norm_scales) == {1, 3, 1}
      assert Nx.shape(outputs.norm_normalized) == {1, 3, 4}
    end

    test "causal LM heads preserve final norm telemetry when opted in" do
      outputs =
        run_qwen3(
          :for_causal_language_modeling,
          output_norm_telemetry: true
        )

      assert Nx.shape(outputs.logits) == {1, 3, 32}
      assert Nx.shape(outputs.norm_scales) == {1, 3, 1}
      assert Nx.shape(outputs.norm_normalized) == {1, 3, 4}
    end
  end

  defp run_gpt2(global_layer_options) do
    run_gpt2(:base, global_layer_options)
  end

  defp run_gpt2(architecture, global_layer_options) do
    spec =
      Bumblebee.configure(Bumblebee.Text.Gpt2,
        architecture: architecture,
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

    input = %{"input_ids" => Nx.tensor([[1, 2, 3]], type: :u32)}
    predict(spec, input, global_layer_options)
  end

  defp run_qwen3(global_layer_options) do
    run_qwen3(:base, global_layer_options)
  end

  defp run_qwen3(architecture, global_layer_options) do
    spec =
      Bumblebee.configure(Bumblebee.Text.Qwen3,
        architecture: architecture,
        vocab_size: 32,
        hidden_size: 4,
        intermediate_size: 8,
        attention_head_size: 2,
        num_blocks: 2,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        max_positions: 8,
        use_qk_norm: false
      )

    input = %{"input_ids" => Nx.tensor([[1, 2, 3]], type: :s64)}
    predict(spec, input, global_layer_options)
  end

  defp predict(spec, input, global_layer_options) do
    model = Bumblebee.build_model(spec)
    {init_fn, predict_fn} = Axon.build(model, global_layer_options: global_layer_options)
    params = init_fn.(Nx.to_template(input), Axon.ModelState.empty())

    predict_fn.(params, input)
  end
end
