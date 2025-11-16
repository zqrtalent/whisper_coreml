import time
import coremltools as ct
import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
import whisper.tokenizer as tk

from whisper.audio import log_mel_spectrogram, pad_or_trim

from whisper import load_model, Whisper as Whisper_real
from whisper.decoding import PyTorchInference as PyTorchInference_real

def exact_div(x, y):
    assert x % y == 0
    return x // y

cu = ct.ComputeUnit.ALL
encoder_mlprogram_path = "out/Whisper_encoder_tiny.mlpackage"
decoder_mlprogram_path = "out/Whisper_decoder_1_tiny.mlpackage"
decoder_3_mlprogram_path = "out/Whisper_decoder_3_tiny.mlpackage"
cross_kv_cache_mlprogram_path = "out/Whisper_cross_kv_cache_tiny.mlpackage"

#ModelDimensions(n_mels=80, n_audio_ctx=1500, n_audio_state=384, n_audio_head=6, n_audio_layer=4, n_vocab=51865, n_text_ctx=448, n_text_state=384, n_text_head=6, n_text_layer=4)
n_state = 384
n_audio_ctx = 1500
n_text_ctx = 448
n_layer = 4
n_head = 6
n_mels = 80
n_vocab=51865
is_multi_lingual = True
num_languages = 99 #self.dims.n_vocab - 51765 - int(self.is_multilingual)

SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000 samples in a 30-second chunk
N_FRAMES = exact_div(N_SAMPLES, HOP_LENGTH)  # 3000 frames in a mel spectrogram input

N_SAMPLES_PER_TOKEN = HOP_LENGTH * 2  # the initial convolutions has stride 2
FRAMES_PER_SECOND = exact_div(SAMPLE_RATE, HOP_LENGTH)  # 10ms per audio frame
TOKENS_PER_SECOND = exact_div(SAMPLE_RATE, N_SAMPLES_PER_TOKEN)  # 20ms per audio token

def encode_audio(m: ct.models.MLModel, audio_file: str):
    mel = log_mel_spectrogram(audio_file, n_mels, padding=N_SAMPLES)
    mel_segment = pad_or_trim(mel, N_FRAMES).to(torch.float32)
    mel_segment = mel_segment.unsqueeze(0)
    
    inp = {
        "x": mel_segment.cpu().numpy()
    }
    
    out = m.predict(inp)
    return out[next(iter(out))] if out is not None else None

def encode_audio_whisper(m: Whisper_real, audio_file: str):
    mel = log_mel_spectrogram(audio_file, n_mels, padding=N_SAMPLES)
    mel_segment = pad_or_trim(mel, N_FRAMES).to(torch.float32)
    mel_segment = mel_segment.unsqueeze(0)
    
    logits = m.encoder(mel_segment)
    return logits

def gen_cross_kv_cache(m: ct.models.MLModel, xa):
    m_cross_kv = ct.models.MLModel(cross_kv_cache_mlprogram_path, compute_units=cu)

    inp_kv = {
        "xa": xa
    }

    kv_out = m_cross_kv.predict(inp_kv)
    cross_cache_k = kv_out["k_cache"] if kv_out is not None else np.zeros((n_layer,1,n_audio_ctx,n_state), np.float32)
    cross_cache_v = kv_out["v_cache"] if kv_out is not None else np.zeros((n_layer,1,n_audio_ctx,n_state), np.float32)
    return cross_cache_k, cross_cache_v

    
def gen_cross_kv_cache_whisper(m: Whisper_real, xa) -> tuple[Tensor, Tensor]:
    """
    Generates a key/value cache based on audio features

    xa - audio features tensor with shape (B,C,S) => (1,1500,384)
    """
    xa = torch.from_numpy(xa)
    k_cache_stack: list[Tensor] = []
    v_cache_stack: list[Tensor] = []
    for block in m.decoder.blocks:
        k = block.cross_attn.key(xa).detach()
        v = block.cross_attn.value(xa).detach()
        k_cache_stack.append(k)
        v_cache_stack.append(v)
    k_cache = torch.stack(k_cache_stack, dim=0)
    v_cache = torch.stack(v_cache_stack, dim=0)
    return (k_cache, v_cache)

def detect_language(m: ct.models.MLModel, audio_features, cross_k_cache, cross_v_cache, return_all = False):
    tokenizer = tk.get_tokenizer(multilingual = is_multi_lingual, num_languages=num_languages)
    
    input_tokens = torch.tensor([tokenizer.sot], dtype=torch.int32)
    logits = torch.from_numpy(decode_sequence(m, input_tokens, audio_features, cross_k_cache, cross_v_cache)[0,:]) # (n_audio, n_vocabsize) => (1,51865)
    n_audio = 1

    # Mask out non language tokens
    mask = torch.ones(logits.shape[-1], dtype=torch.bool)
    mask[list(tokenizer.all_language_tokens)] = False
    logits[:, mask] = -np.inf
    language_tokens = logits.argmax(dim=-1)
    language_token_probs = logits.softmax(dim=-1).cpu()
    language_probs = [
        {
            c: language_token_probs[i, j].item()
            for j, c in zip(tokenizer.all_language_tokens, tokenizer.all_language_codes)
        }
        for i in range(n_audio)
    ]

    if not return_all:
        language_tokens = language_tokens[0]
        language_probs = language_probs[0]

    return language_tokens, language_probs

def decode_sequence(m: ct.models.MLModel, input_text_tokens, audio_features, cross_k_cache, cross_v_cache, pos: int, state: ct.models.model.MLState) -> np.ndarray:
    tokens = np.array([input_text_tokens], dtype=np.int32)
    
    # Only one token is accepted by model.
    pos = np.array([pos], dtype=np.int32)

    inp = {
        "x": tokens,
        "xa": audio_features,
        "pos": pos,
        "cross_cache_k_1": cross_k_cache[0],
        "cross_cache_v_1": cross_v_cache[0],
        "cross_cache_k_2": cross_k_cache[1],
        "cross_cache_v_2": cross_v_cache[1],
        "cross_cache_k_3": cross_k_cache[2],
        "cross_cache_v_3": cross_v_cache[2],
        "cross_cache_k_4": cross_k_cache[3],
        "cross_cache_v_4": cross_v_cache[3],
    }

    if state is None:
        state = m.make_state()
    out = m.predict(inp, state)
    return out["logits"]

def decode_sequence_whisper(m: PyTorchInference_real, input_text_tokens, audio_features):
    result = m.logits(torch.from_numpy(input_text_tokens), torch.from_numpy(audio_features))
    return result

def compare_results(ml_result, whisper_result):
    t_pt = whisper_result
    t_ml = torch.from_numpy(ml_result)
    if t_pt.shape != t_ml.shape:
        print(f"shape are not the same! ml {ml_result.shape} pt {t_pt.shape}")
        return False

    diff = t_pt - t_ml
    mae = diff.abs().mean()
    max_err = diff.abs().max()
    l2 = torch.norm(diff) / torch.norm(t_pt)
    cos = torch.nn.functional.cosine_similarity(t_pt, t_ml, dim=-1).mean()
    print(f"ml -> pt")
    print(f"mean = {mae}")
    print(f"max_err = {max_err}")
    print(f"l2 = {l2}")
    print(f"cos = {cos}")
    
def compare_decoder_results(decoder_result_ml, decoder_result_whisper):
    print("Comparing results")
    cmp_tensor = torch.from_numpy(decoder_result_ml).allclose(decoder_result_whisper, atol=2e-03, rtol=0)
    print(f"All close {cmp_tensor}")
    compare_results(decoder_result_ml, decoder_result_whisper)
    
def test_model(audio_file: str, model_name: str):
    model = load_model(model_name)
    
    # Encode audio file
    print("Running encoder using coreml.")
    m_encoder = ct.models.MLModel(encoder_mlprogram_path, compute_units=cu) 
    audio_features = encode_audio(m_encoder, audio_file)
    print(audio_features.shape)
    print("=================")

    # Whisper encoder to generate audio features.
    print("Running encoder using whisper.")
    whisper_audio_features = encode_audio_whisper(model, audio_file)
    print("Comparing results")
    cmp_tensor = torch.from_numpy(audio_features).allclose(whisper_audio_features, atol=2e-03, rtol=0)
    print(f"All close {cmp_tensor}")
    compare_results(audio_features, whisper_audio_features)
    print("=================")
    
    # Calculate kv of the audio features
    print("Running encoder using whisper.")
    m_cross_kv = ct.models.MLModel(cross_kv_cache_mlprogram_path, compute_units=cu)
    cross_cache_k, cross_cache_v = gen_cross_kv_cache(m_cross_kv, audio_features)
    print(f"cross_cache_k={cross_cache_k.shape} cross_cache_v={cross_cache_v.shape}")
    print("===========================")
    
    print("Running cross kv cache using whisper.")
    cross_cache_k_whisper, cross_cache_v_whisper = gen_cross_kv_cache_whisper(model, audio_features)
    print("Comparing results")
    cmp_tensor = torch.from_numpy(cross_cache_k).allclose(cross_cache_k_whisper, atol=2e-03, rtol=0)
    print(f"All close (cross_cache_k) {cmp_tensor}")
    cmp_tensor = torch.from_numpy(cross_cache_v).allclose(cross_cache_v_whisper, atol=2e-03, rtol=0)
    print(f"All close (cross_cache_v) {cmp_tensor}")
    print("cross_cache_k:")
    compare_results(cross_cache_k, cross_cache_k_whisper)
    print("cross_cache_v:")
    compare_results(cross_cache_v, cross_cache_v_whisper)
    print("===========================")
    
    # test_decoder_use_case_1(model, audio_features, cross_cache_k, cross_cache_v)
    test_decoder_use_case_2(model, audio_features, cross_cache_k, cross_cache_v)
    
def test_decoder_use_case_1(model: Whisper_real, audio_features, cross_cache_k, cross_cache_v):
    """
    Test use case 1: decoder is tasked to predict sequence 2 times (one token each time).
    """
    print("Test decoder use case #1.")
    # Run decoder
    print("Running decoder using coreml.")
    # [50258,50259,50359]
    tokens = [50258] 
    m_decoder = ct.models.MLModel(decoder_mlprogram_path, compute_units=cu)
    state = m_decoder.make_state()
    decoder_result_ml = decode_sequence(m_decoder, tokens, audio_features, cross_cache_k, cross_cache_v, pos=0, state=state)
    print(decoder_result_ml.shape)
    print("=================")
    
    print("Running decoder using whisper.")
    inference = PyTorchInference_real(model, initial_token_length=0)
    decoder_result_whisper = decode_sequence_whisper(inference, np.array([tokens]), audio_features)
    compare_decoder_results(decoder_result_ml, decoder_result_whisper)
    print("=================")
    
    print("X2 Running decoder using coreml.")
    tokens = [50258,50259]
    decoder_result_ml = decode_sequence(m_decoder, tokens[1:], audio_features, cross_cache_k, cross_cache_v, pos=1, state=state)
    print(decoder_result_ml.shape)
    print("=================")
    
    print("X2 Running decoder using whisper.")
    decoder_result_whisper = decode_sequence_whisper(inference, np.array([tokens]), audio_features)
    compare_decoder_results(decoder_result_ml, decoder_result_whisper)
    print("=================")
    
def test_decoder_use_case_2(model: Whisper_real, audio_features, cross_cache_k, cross_cache_v):
    """
    Test use case 1: decoder is tasked to predict sequence 2 times. first 3 tokens and the 1
    """
    print("Test decoder use case #2.")
    # Run decoder
    print("Running decoder using coreml.")
    tokens = [50258,50259,50359]
    m_decoder_3 = ct.models.MLModel(decoder_3_mlprogram_path, compute_units=cu)
    m_decoder_1 = ct.models.MLModel(decoder_mlprogram_path, compute_units=cu)
    state = m_decoder_3.make_state()
    decoder_result_ml = decode_sequence(m_decoder_3, tokens, audio_features, cross_cache_k, cross_cache_v, pos=0, state=state)
    print(decoder_result_ml.shape)
    print("=================\r\n")
    
    print("Running decoder using whisper.")
    inference = PyTorchInference_real(model, initial_token_length=3)
    decoder_result_whisper = decode_sequence_whisper(inference, np.array([tokens]), audio_features)
    compare_decoder_results(decoder_result_ml, decoder_result_whisper)
    print("Results:")
    print(f"coreml = {torch.from_numpy(decoder_result_ml)[0,-1,:].view(-1).argmax(dim=-1)} torch = {decoder_result_whisper[0,-1,:].view(-1).argmax(dim=-1)}")
    print("=================\r\n")
    
    next_tokens = [50363, 2425, 456, 11, 7751, 456, 13, 50564]
    for i, token in enumerate(next_tokens):
        print(f"X{i + 1} Running decoder using coreml.")
        tokens.append(token)
        decoder_result_ml = decode_sequence(m_decoder_1, [tokens[-1]], audio_features, cross_cache_k, cross_cache_v, pos=3 + i, state=state)
        print(decoder_result_ml.shape)
        print("=================\r\n")
        
        print(f"X{i+1} Running decoder using whisper.")
        decoder_result_whisper = decode_sequence_whisper(inference, np.array([tokens]), audio_features)
        compare_decoder_results(decoder_result_ml, decoder_result_whisper)
        print("Results:")
        print(f"coreml = {torch.from_numpy(decoder_result_ml).view(-1).argmax(dim=-1)} torch = {decoder_result_whisper.view(-1).argmax(dim=-1)}")
        print("=================\r\n")
    
def transcribe(audio_file: str):
     # Encode audio file
    m_encoder = ct.models.MLModel(encoder_mlprogram_path, compute_units=cu) 
    audio_features = encode_audio(m_encoder, audio_file)

    # Calculate kv of the audio features
    m_cross_kv = ct.models.MLModel(cross_kv_cache_mlprogram_path, compute_units=cu)
    cross_cache_k, cross_cache_v = gen_cross_kv_cache(m_cross_kv, audio_features)
    
    # Greedy decoder (simplest version)
    _eot = 50257
    _sot = 50258
    _transcribe = 50359
    _en = 50259
    _space = 220
    _no_speech = 50362
    _begin_sample = [_sot, _en, _transcribe]
    _suppress_tokens = [1,2,7,8,9,10,14,25,26,27,28,29,31,58,59,60,61,62,63,90,91,92,93,359,503,522,542,873,893,902,918,922,931,1350,1853,1982,2460,2627,3246,3253,3268,3536,3846,3961,4183,4667,6585,6647,7273,9061,9383,10428,10929,11938,12033,12331,12562,13793,14157,14635,15265,15618,16553,16604,18362,18956,20075,21675,22520,26130,26161,26435,28279,29464,31650,32302,32470,36865,42863,47425,49870,50254,50258,50358,50359,50360,50361,50362]
    _no_timestamps = 50363
    _sot_index = 0
     
    n_batch = 1
    
    def apply_filter(tokens: Tensor, logits: Tensor):
        # suppress blank
        if tokens.shape[-1] == len(_begin_sample):
            logits[:, [_space,_eot]] = float('-inf')
        # suppress special tokens
        logits[:, _suppress_tokens] = float('-inf')
        # apply timestamp rules
        # suppress <|notimestamps|> which is handled by without_timestamps
        # if self.tokenizer.no_timestamps is not None:
        logits[:, _no_timestamps] = -np.inf
    
    def greedy_decode(tokens:Tensor, logits: Tensor, sum_logprobs: Tensor) -> tuple[Tensor,bool]:
        logits = logits.float()
        next_tokens = logits.argmax(dim=-1)
        # if self.temperature == 0:
        #     next_tokens = logits.argmax(dim=-1)
        # else:
        #     next_tokens = Categorical(logits=logits / self.temperature).sample()
        
        logprobs = F.log_softmax(logits.float(), dim=-1)
        current_logprobs = logprobs[torch.arange(logprobs.shape[0]), next_tokens]
        sum_logprobs += current_logprobs * (tokens[:, -1] != _eot)

        next_tokens[tokens[:, -1] == _eot] = _eot
        tokens = torch.cat([tokens, next_tokens[:, None]], dim=-1)
        
        completed = (tokens[:, -1] == _eot).all()
        return tokens, completed
    
    
    loop = 0
    pos = 0
    max_tokens = n_text_ctx/2
    tokens = torch.tensor([[_sot, _en, _transcribe]], dtype=torch.int32) #
    m_decoder_1: ct.models.MLModel = ct.models.MLModel(decoder_mlprogram_path, compute_units=cu)
    m_decoder_3: ct.models.MLModel = ct.models.MLModel(decoder_3_mlprogram_path, compute_units=cu)
    state = m_decoder_1.make_state()
    
    sum_logprobs: Tensor = torch.zeros((n_batch), dtype=torch.float32)
    no_speech_probs = [np.nan] * n_batch
    
    start_perf = time.perf_counter()
    while loop < max_tokens:
        # Decode sequence method only support single batch.
        logits = decode_sequence(m_decoder_3 if loop == 0 else m_decoder_1, tokens.view(-1).tolist() if loop == 0 else [tokens[0,-1]], audio_features, cross_cache_k, cross_cache_v, pos, state)
        logits = torch.from_numpy(logits)
        
        if loop == 0:  # save no_speech_probs
            probs_at_sot = logits[:, _sot_index].softmax(dim=-1)
            no_speech_probs = probs_at_sot[:, _no_speech].tolist()
            
        logits = logits[:, -1] # Consider last token logits only.
        pos = pos + (tokens.shape[-1] if loop == 0 else 1)
        
        apply_filter(tokens, logits)
        
        tokens, completed = greedy_decode(tokens, logits, sum_logprobs)
        
        if completed or tokens.shape[-1] > n_text_ctx:
            break
        loop = loop + 1
        
    tokenizer = tk.get_tokenizer(multilingual = is_multi_lingual, num_languages=num_languages)
    text = tokenizer.decode(tokens.view(-1).tolist())
    print(text)
    end_perf = time.perf_counter()
    print(f"took {end_perf - start_perf:0.4f} seconds")
    
    # tokens:
    # eot - 50257
    # sot - 50258
    
    # Decode sequence
    # Iterate over 30s segments
    # generate mel spectogram of segment
    # [encoder model] generate audio features
    # [decoder kv cache model] generate kv cache from audio features
    # [decoder model] decode with fallback
    #   
    # Check for no speech 
    # initial tokens: 50258 50259 50359
    # loop through 0...448/2
    #   first loop:
    #   get logits for  [50258,50259,50359]
    #   get sot token probabilities and check check no speach (50362) probability.
    #   retrieve last tokens logits only logits[:,-1]
    #   apply logit filters:  
    #       suppress blank only for initial tokens (first loop): set tokens of " "(220) and eot (50257) as -inf
    #       suppress tokens: 1,2,7,8,9,10,14,25,26,27,28,29,31,58,59,60,61,62,63,90,91,92,93,359,503,522,542,873,893,902,918,922,931,1350,1853,1982,2460,2627,3246,3253,3268,3536,3846,3961,4183,4667,6585,6647,7273,9061,9383,10428,10929,11938,12033,12331,12562,13793,14157,14635,15265,15618,16553,16604,18362,18956,20075,21675,22520,26130,26161,26435,28279,29464,31650,32302,32470,36865,42863,47425,49870,50254,50258,50358,50359,50360,50361,50362
    #       suppress no timestamp: no_timestamp (50363), timestamp_begin (50364) only for the first loop, all after 50414 [:, 50415: ]
    #   update decoder (Greedy decoder):
    #       if temparature is 0 -> retrieve an argmax from logits 'logits.argmax(dim=-1)' else 'Categorical(logits=logits / self.temperature).sample()' 
    #       use result as next_tokens.
    #       calcualate log probs from logits
    #       retrieve current log probs of next tokens
    #       no idea => 'next_tokens[tokens[:, -1] == self.eot] = self.eot'
    #       append new token 'next_tokens' to the tokens. `tokens = torch.cat([tokens, next_tokens[:, None]], dim=-1)`
    #       determine if completed: `completed = (tokens[:, -1] == self.eot).all()`
    #       return tokens, completed, sum_log_probs
    #   break if completed flag is received or tokens exceed text max context. (448)
    # append tokens with eot (50257).
    # cut only valid tokens from tokens tensor. Ignore beginning 3 tokens and ending eot.
    # Maximum likelyhood sample ranker logic.
    
import argparse

if __name__ == "__main__":
    audio_file = "./audio/sample.m4a"
    # audio_file = "./audio/Saitama vs Genos Fight  One Punch Man.mp3"
    # audio_file = "./audio/Formula News.mp3"
    model_name = "tiny"
    
    parser = argparse.ArgumentParser(prog="RunCoreML", description="Runs a whisper coreml model. ONLY works on MacOS 15+)")
    parser.add_argument("--run_test", type=bool, required=False)
    parser.add_argument("--run_transcribe", type=bool, required=False)
    args = parser.parse_args()
    
    print(args)
    run_test = args.run_test if args.run_test is not None else True
    if run_test:
        test_model(audio_file, model_name)
    run_transcribe = args.run_transcribe if args.run_transcribe is not None else False
    if run_transcribe:
        transcribe(audio_file)