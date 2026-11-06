"""单卡冒烟测试 1：vLLM 加载 Qwen2.5-0.5B-Instruct，用聊天模板生成两个简短回答。"""
from vllm import LLM, SamplingParams

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
QUESTIONS = [
    ("What is the capital of Sweden? Answer in one word.", "stockholm"),
    ("What is 12 * 7? Answer with the number only.", "84"),
]


def main():
    llm = LLM(model=MODEL, gpu_memory_utilization=0.3, max_model_len=2048, enforce_eager=True, seed=0)
    tok = llm.get_tokenizer()
    prompts = [
        tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
        for q, _ in QUESTIONS
    ]
    outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=16))
    texts = [o.outputs[0].text.strip() for o in outs]

    for (q, expect), t in zip(QUESTIONS, texts):
        mark = "ok" if expect in t.lower() else "（答案和预期不同，但只要非空就算通过）"
        print(f"  Q: {q}\n  A: {t}  {mark}")
    assert all(texts), "vLLM 生成了空字符串"
    print("GPU1 VLLM PASS")


if __name__ == "__main__":  # vLLM 会启动子进程，必须有这个保护
    main()
