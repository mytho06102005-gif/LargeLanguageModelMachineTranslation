# LargeLanguageModelMachineTranslation
- Mô hình sau khi train xong được up lên Hugging face: https://huggingface.co/Mytho0610/LLMMachineTranslation
- Link tải dataset: https://huggingface.co/datasets/thainq107/iwslt2015-en-vi

Sau khi huấn luyện, mô hình được lưu trữ trên nền tảng Hugging Face nhằm hỗ trợ tái sử dụng và triển khai mà không cần thực hiện huấn luyện lại. Người dùng có thể tải trực tiếp mô hình đã huấn luyện để dịch văn bản mới. Cách load lại mô hình để tiếp tục dịch:
Bước 1:
```python
!pip install transformers accelerate sentencepiece gradio -q
```
Bước 2:
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

model_id = "Mytho0610/LLMMachineTranslation"

print("Đang load model...")

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

print("Load model thành công!")
```
Bước 3:
```python
import gradio as gr
def translate(sentence, max_new_tokens=100):

    prompt = f"""
### Instruction:
Translate the following English sentence into Vietnamese.

### English:
{sentence}

### Vietnamese:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            do_sample=False,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    result = decoded.split("### Vietnamese:")[-1]

    # Xóa phần sinh dư
    stop_words = [
        "###",
        "## Translation",
        "Translation Memory"
    ]

    for stop in stop_words:
        result = result.split(stop)[0]

    result = result.strip()

    return result

with gr.Blocks(title="Dịch Anh → Việt") as demo:

    gr.Markdown("## 🌐 Dịch Tiếng Anh → Tiếng Việt")

    with gr.Row():

        with gr.Column():

            input_text = gr.Textbox(
                label="Tiếng Anh",
                placeholder="Nhập câu tiếng Anh...",
                lines=5
            )

            with gr.Row():
                btn_translate = gr.Button("Dịch", variant="primary")
                btn_clear = gr.Button("Xóa")

        with gr.Column():

            output_text = gr.Textbox(
                label="Tiếng Việt",
                lines=5,
                interactive=False
            )

    btn_translate.click(
        fn=translate,
        inputs=input_text,
        outputs=output_text
    )

    input_text.submit(
        fn=translate,
        inputs=input_text,
        outputs=output_text
    )

    btn_clear.click(
        fn=lambda: ("", ""),
        inputs=None,
        outputs=[input_text, output_text]
    )

demo.launch(share=True)
```
   
