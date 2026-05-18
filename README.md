# LargeLanguageModelMachineTranslation
- Mô hình sau khi train xong được up lên Hugging face: https://huggingface.co/Mytho0610/LLMMachineTranslation
- Link tải dataset: https://huggingface.co/datasets/thainq107/iwslt2015-en-vi

Sau khi huấn luyện, mô hình được lưu trữ trên nền tảng Hugging Face nhằm hỗ trợ tái sử dụng và triển khai mà không cần thực hiện huấn luyện lại. Người dùng có thể tải trực tiếp mô hình đã huấn luyện để dịch văn bản mới. Cách load lại mô hình để tiếp tục dịch:
```python
import torch
import gradio as gr

# Hàm dịch
def translate(sentence, max_len=50):

    model.eval()

    tokens = tokenizer.encode(sentence).ids

    src = torch.LongTensor(tokens).unsqueeze(0).to(device)

    with torch.no_grad():

        output_ids = model.generate(
            src,
            max_length=max_len
        )

    result = tokenizer.decode(output_ids[0].cpu().numpy())

    return result


# Giao diện Gradio
with gr.Blocks(title="Dịch Anh → Việt") as demo:

    gr.Markdown("## 🌐 Dịch Tiếng Anh → Tiếng Việt")
    gr.Markdown("Model: Mytho0610/LLMMachineTranslation")

    with gr.Row():

        with gr.Column():

            input_text = gr.Textbox(
                label="Tiếng Anh",
                placeholder="Nhập câu tiếng Anh...",
                lines=5
            )

            btn_translate = gr.Button("Dịch", variant="primary")
            btn_clear = gr.Button("Xóa")

        with gr.Column():

            output_text = gr.Textbox(
                label="Tiếng Việt",
                lines=5,
                interactive=False
            )

    # Nút dịch
    btn_translate.click(
        fn=translate,
        inputs=input_text,
        outputs=output_text
    )

    # Enter để dịch
    input_text.submit(
        fn=translate,
        inputs=input_text,
        outputs=output_text
    )

    # Xóa
    btn_clear.click(
        fn=lambda: ("", ""),
        inputs=None,
        outputs=[input_text, output_text]
    )

demo.launch(share=True)
```
