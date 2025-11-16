import os
import torch
import numpy as np
import coremltools as ct

from whisper import Whisper, load_model
from coremltools.converters.mil.mil import Program
from whisper.model import Whisper_CrossKV_Generator

# import whisper_real.whisper as w

def convert_cross_state_cache_model(model: "Whisper", model_type: str = "tiny", out_dir: str = "out") -> any:
    generator_model = Whisper_CrossKV_Generator(model.eval())
    # (B,C,S) => (1, 1500,384)
    xa = torch.randn((1, model.dims.n_audio_ctx, model.dims.n_audio_state))

    traced_model = torch.jit.trace(generator_model, (xa), check_trace=True)

    mlmodel = ct.convert(traced_model,
                         inputs=[
                             ct.TensorType(name="xa", shape=xa.shape)# (1,1500,384) (B,C,S)
                         ],
                         outputs=[
                             ct.TensorType(name="k_cache"), # (4,1,1500,384) (L,B,C,S)
                             ct.TensorType(name="v_cache"), # (4,1,1500,384) (L,B,C,S)
                         ],
                         convert_to="mlprogram",
                         compute_units=ct.ComputeUnit.ALL,
                         minimum_deployment_target=ct.target.iOS18,
                         compute_precision=ct.precision.FLOAT32,
                         # skip_model_load=True,
                         debug=True
                         )

    mlmodel.save(f"{out_dir}/Whisper_cross_kv_cache_{model_type}.mlpackage")
    with open(f"{out_dir}/Whisper_cross_kv_cache_{model_type}.mil", "w") as f:
        f.write(f"{mlmodel._mil_program}")
    return mlmodel


def convert_encoder_model(model: "Whisper", model_type: str = "tiny", out_dir: str = "out") -> any:
    encoder_model = model.encoder
    encoder_model.eval()
    audio_mel = torch.randn((1, model.dims.n_mels, 3000),
                            dtype=torch.float32)  # (B,80,3000)
    # encoder_model(audio_mel)

    traced_encoder_model = torch.jit.trace(
        encoder_model, (audio_mel), check_trace=True)

    mlmodel = ct.convert(traced_encoder_model,
                         inputs=[ct.TensorType(
                             name="x", shape=audio_mel.shape, dtype=np.float32)],
                         # (1,1500,384) (B,C,S)
                         outputs=[ct.TensorType(
                             name="audio_features", dtype=np.float32)],
                         convert_to="mlprogram",
                         compute_units=ct.ComputeUnit.ALL,
                         minimum_deployment_target=ct.target.iOS18,
                         compute_precision=ct.precision.FLOAT32,
                         )

    print("=======================")
    print(mlmodel._mil_program)

    mlmodel.save(f"{out_dir}/Whisper_encoder_{model_type}.mlpackage")

    with open(f"{out_dir}/Whisper_encoder_{model_type}.mil", "w") as f:
        f.write(f"{mlmodel._mil_program}")
    return mlmodel


def convert_decoder_model(model: "Whisper", model_type: str = "tiny", out_dir: str = "out", sequence_len: int = 1) -> any:
    dims = model.dims
    decoder_model = model.decoder
    decoder_model.eval()

    batch_size = 1
    tokens = [50258, 50259, 50359]
    audio_features = torch.randn((batch_size, model.dims.n_audio_ctx, model.dims.n_audio_state),
                                 # (B,1500,384)
                                 dtype=torch.float32, device=model.device)
    text_tokens = torch.tensor(
        tokens[:sequence_len], dtype=torch.int32, device=model.device).view(1, -1)

    # Generate and update cross attn k/v cache.
    k_cache, v_cache = decoder_model.audio_features_kv(audio_features)

    n_layer = model.dims.n_text_layer
    n_head = model.dims.n_text_head
    n_dim = model.dims.n_audio_state  # embeddings dimension

    pos = torch.tensor([0], dtype=torch.long)

    # logits = decoder_model(text_tokens, audio_features, pos,
    #                                                k_cache[0], v_cache[0], k_cache[1], v_cache[1],
    #                                                k_cache[2], v_cache[2], k_cache[3], v_cache[3])

    # inference_tokens = torch.tensor(tokens[:3], dtype=torch.int32, device=model.device).view(1,-1)
    # model_real = w.load_model(model_type)
    # inference = w.decoding.PyTorchInference(model_real, 3)
    # logits_real = inference.logits(inference_tokens, audio_features)
    # cmp_result = torch.allclose(logits, logits_real, atol=1e-3, rtol=0)
    # print(f"compare result for sequence 1: {cmp_result}")

    # pos = pos + 1
    # text_tokens = torch.tensor(tokens[1:2], dtype=torch.int32, device=model.device).view(1,-1)
    # logits_2 = decoder_model(text_tokens, audio_features, pos,
    #                                                k_cache[0], v_cache[0], k_cache[1], v_cache[1],
    #                                                k_cache[2], v_cache[2], k_cache[3], v_cache[3]).view(-1)
    # inference_tokens = torch.tensor(tokens[:2], dtype=torch.int32, device=model.device).view(1,-1)
    # logits_real_2 = inference.logits(inference_tokens, audio_features).view(-1)
    # cmp_result2 = torch.allclose(logits_2, logits_real_2, atol=1e-3, rtol=0)
    # print(f"compare result for sequence 2: {cmp_result2}")
    # return

    # # boolean mask of differences
    # diff_mask = (logits_real - logits).abs() > 0.01

    # # indices where they differ
    # diff_indices = diff_mask.nonzero(as_tuple=False)

    # # paired values for inspection
    # diff_values = [(logits_real[i].item(), logits[i].item()) for i in diff_indices]

    # print(diff_indices)
    # print(diff_values)

    traced_model = torch.jit.trace(decoder_model, (text_tokens, audio_features, pos,
                                                   k_cache[0], v_cache[0], k_cache[1], v_cache[1],
                                                   k_cache[2], v_cache[2], k_cache[3], v_cache[3]), check_trace=False)

    # rangeDimState = ct.RangeDim(lower_bound=1, upper_bound=model.dims.n_text_ctx, default=model.dims.n_text_ctx)
    # rangeDimInputCross = ct.RangeDim(lower_bound=1, upper_bound=model.dims.n_audio_ctx, default=1)
    # rangeDimInput = ct.RangeDim(lower_bound=0, upper_bound=model.dims.n_text_ctx)

    state_shape = (n_layer, batch_size, model.dims.n_text_ctx, n_dim)
    cross_cache_shape = (batch_size, model.dims.n_audio_ctx, n_dim)

    mlmodel = ct.convert(traced_model,
                         inputs=[
                             ct.TensorType(name="x", shape=text_tokens.shape, dtype=np.int32),
                             ct.TensorType(name="xa", shape=audio_features.shape, dtype=np.float32),
                             ct.TensorType(name="pos", shape=(1,), dtype=np.long),
                             ct.TensorType(name="cross_cache_k_1", shape=cross_cache_shape, dtype=np.float32),
                             ct.TensorType(name="cross_cache_v_1", shape=cross_cache_shape, dtype=np.float32),
                             ct.TensorType(name="cross_cache_k_2", shape=cross_cache_shape, dtype=np.float32),
                             ct.TensorType(name="cross_cache_v_2", shape=cross_cache_shape, dtype=np.float32),
                             ct.TensorType(name="cross_cache_k_3", shape=cross_cache_shape, dtype=np.float32),
                             ct.TensorType(name="cross_cache_v_3", shape=cross_cache_shape, dtype=np.float32),
                             ct.TensorType(name="cross_cache_k_4", shape=cross_cache_shape, dtype=np.float32),
                             ct.TensorType(name="cross_cache_v_4", shape=cross_cache_shape, dtype=np.float32),
                         ],
                         outputs=[
                             ct.TensorType(name="logits"),
                         ],
                         states=[
                             ct.StateType(name="selfKeyCache", wrapped_type=ct.TensorType(shape=state_shape, dtype=np.float16)),
                             ct.StateType(name="selfValueCache", wrapped_type=ct.TensorType(shape=state_shape, dtype=np.float16)),
                         ],
                         convert_to="mlprogram",
                         compute_units=ct.ComputeUnit.ALL,
                         minimum_deployment_target=ct.target.iOS18,
                         compute_precision=ct.precision.FLOAT32,
                         # skip_model_load=True,
                         debug=True
                         )

    print("=======================")
    print(mlmodel._mil_program)

    file_name = f"Whisper_decoder_{sequence_len}_{model_type}"
    model_file_path = f"{out_dir}/{file_name}.mlpackage"
    # if os.path.exists(model_file_path):
    #     os.remove(model_file_path)
    mlmodel.save(model_file_path)
    with open(f"{out_dir}/{file_name}.mil", "w") as f:
        f.write(f"{mlmodel._mil_program}")
    return mlmodel


def main():
    out_dir = "out"
    model_type = "tiny"
    
    print("loading a model:", model_type)
    model = load_model(model_type, is_coreml_model=True, token_seq_len=1)

    print(model.dims)
    print("model loaded.")

    print("converting an encoder model.")
    convert_encoder_model(model, model_type=model_type, out_dir=out_dir)
    print("encoder model converted.")

    print("converting a decoder model. fixed sequence of 1")
    convert_decoder_model(model, model_type=model_type,
                          out_dir=out_dir, sequence_len=1)
    print("decoder model converted.")

    print("converting a decoder model. fixed sequence of 3")
    model3 = load_model(model_type, is_coreml_model=True, token_seq_len=3)
    convert_decoder_model(model3, model_type=model_type,
                          out_dir=out_dir, sequence_len=3)
    print("decoder model converted.")

    print("converting a cross_kv cache model.")
    convert_cross_state_cache_model(model, model_type=model_type, out_dir=out_dir )
    print("cross_kv cache model converted.")


if __name__ == "__main__":
    main()
