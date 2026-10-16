"""Isolated CUDA text inference for larger v2 research candidates."""
import time


class HFBackend:
    def __init__(self,model_path,max_input_tokens=4096,load_in_4bit=False):
        import torch
        from transformers import AutoConfig,AutoModelForCausalLM,AutoModelForImageTextToText,AutoTokenizer
        if not torch.cuda.is_available():
            raise RuntimeError('The remote v2 experiment requires a Slurm GPU allocation')
        start=time.perf_counter()
        config=AutoConfig.from_pretrained(model_path,trust_remote_code=False)
        cls=AutoModelForImageTextToText if config.model_type.startswith('qwen3_5') else AutoModelForCausalLM
        self.tokenizer=AutoTokenizer.from_pretrained(model_path,trust_remote_code=False)
        kwargs={'dtype':torch.bfloat16,'attn_implementation':'sdpa','trust_remote_code':False}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs['quantization_config']=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
                    bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16)
            kwargs['device_map']={'':0}
        self.model=cls.from_pretrained(model_path,**kwargs).eval()
        if not load_in_4bit:
            self.model=self.model.to('cuda')
        self.load_in_4bit=load_in_4bit
        self.max_input_tokens=max_input_tokens
        self.load_seconds=time.perf_counter()-start
        self.model_path=str(model_path)
        eos=self.model.generation_config.eos_token_id
        self.eos=set(eos if isinstance(eos,list) else [eos])
        self.eos.add(self.tokenizer.eos_token_id)

    def __call__(self,messages,max_tokens=384):
        import torch
        if max_tokens<=0:
            raise ValueError('max_tokens must be positive')
        prompt=self.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        inputs=self.tokenizer(prompt,return_tensors='pt',add_special_tokens=False).to('cuda')
        count=inputs['input_ids'].shape[-1]
        if count>self.max_input_tokens:
            raise ValueError(f'Input {count} exceeds {self.max_input_tokens}; no truncation')
        torch.cuda.synchronize();start=time.perf_counter()
        with torch.inference_mode():
            generated=self.model.generate(**inputs,max_new_tokens=max_tokens,do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
        torch.cuda.synchronize()
        ids=generated[0,count:].tolist()
        return {'text':self.tokenizer.decode(ids,skip_special_tokens=True).strip(),
                'seconds':time.perf_counter()-start,'input_tokens':count,'generation_tokens':len(ids),
                'finish_reason':'stop' if ids and ids[-1] in self.eos else 'length'}
